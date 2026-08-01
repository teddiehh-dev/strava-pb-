import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import differential_evolution
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ CONFIG
# ==========================================
MIN_DISTANCE_KM = 4.8
EARLIEST_ACTIVITY_DATE = '2025-01-01'
QUALITY_WEEK_PACE_MULTIPLIER = 1.25

RIEGEL_GENERIC_EXPONENT = 1.06
MIN_RUNS_FOR_PERSONAL_EXPONENT = 10
MIN_DISTANCE_SPREAD_RATIO = 1.5
PERSONAL_EXPONENT_BOUNDS = (1.00, 1.15)

RECENCY_HALF_LIFE_DAYS = 180
RECENCY_WEIGHT_FLOOR = 0.35
TEMP_RELIABILITY_FLOOR = 0.5

MIN_ROWS_FOR_TUNING = 15
EXTRAPOLATION_MARGIN = 0.10
DENSITY_PENALTY_WEIGHT = 25.0

TARGET_DISTANCES_KM = {'5K': 5.0, '10K': 10.0}
QUANTILES = {'low': 0.2, 'median': 0.5, 'high': 0.8}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def format_time(sec):
    if pd.isna(sec) or sec <= 0:
        return "N/A"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def safe_parse_time(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip()
    try:
        if ':' in v:
            parts = [float(p) for p in v.split(':')]
            return parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else parts[0] * 60 + parts[1]
        return float(v.replace(',', ''))
    except Exception:
        return np.nan

def safe_numeric(df, col, default=np.nan):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype='float64')
    return pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

def recency_weights(dates, reference_date, half_life_days=RECENCY_HALF_LIFE_DAYS, floor=RECENCY_WEIGHT_FLOOR):
    dates = pd.DatetimeIndex(dates)
    days_ago = np.clip((reference_date - dates).days.astype(float), 0, None)
    return floor + (1 - floor) * np.exp(-np.log(2) / half_life_days * days_ago)

# ==========================================
# 1. LOAD AND CLEAN
# ==========================================
def load_and_clean(df):
    df.columns = df.columns.str.strip()

    if 'Activity Type' in df.columns:
        df = df[df['Activity Type'].isin(['Run', 'Trail Run', 'Virtual Run'])].copy()

    df['date'] = pd.to_datetime(df['Activity Date'], format='mixed')
    df = df.sort_values('date')
    if EARLIEST_ACTIVITY_DATE:
        df = df[df['date'] >= pd.Timestamp(EARLIEST_ACTIVITY_DATE)]

    df['moving_time_sec'] = (df['Moving Time'] if 'Moving Time' in df.columns
                              else df.get('Elapsed Time', np.nan)).apply(safe_parse_time)
    dist_raw = safe_numeric(df, 'Distance')
    df['distance_km'] = dist_raw / 1000.0 if dist_raw.max() > 500 else dist_raw
    df = df.dropna(subset=['distance_km', 'moving_time_sec'])
    df['pace_sec_km'] = df['moving_time_sec'] / df['distance_km']

    df['hr'] = safe_numeric(df, 'Average Heart Rate')
    hr_available = df['hr'].notna().sum() >= 4

    device_temp = safe_numeric(df, 'Average Temperature')
    weather_temp = safe_numeric(df, 'Weather Temperature')
    df['temp'] = weather_temp.fillna(device_temp)
    df['temp_reliable'] = weather_temp.notna().astype(float)

    df['elevation_m'] = safe_numeric(df, 'Elevation Gain', 0.0).fillna(0.0)
    df['training_load'] = safe_numeric(df, 'Training Load')
    effort_available = (not hr_available) and (df['training_load'].notna().sum() >= 4)

    qualifying = df[df['distance_km'] >= MIN_DISTANCE_KM]
    gap_days = qualifying['date'].diff().dt.days.dropna()
    n_blocks = 1 + (gap_days > 21).sum() if len(gap_days) else 1

    report = {
        'total_runs': len(qualifying),
        'min_dist': qualifying['distance_km'].min(),
        'max_dist': qualifying['distance_km'].max(),
        'start_date': qualifying['date'].min().date() if len(qualifying) else None,
        'end_date': qualifying['date'].max().date() if len(qualifying) else None,
        'blocks': n_blocks,
        'hr_runs': df['hr'].notna().sum(),
        'load_runs': df['training_load'].notna().sum(),
        'weather_runs': weather_temp.notna().sum(),
        'device_runs': (weather_temp.isna() & device_temp.notna()).sum()
    }

    return df, hr_available, effort_available, report

# ==========================================
# 2. PERSONAL RIEGEL EXPONENT
# ==========================================
def fit_personal_riegel_exponent(qualifying_df):
    if len(qualifying_df) == 0:
        return RIEGEL_GENERIC_EXPONENT, "No qualifying runs found."
        
    spread = qualifying_df['distance_km'].max() / qualifying_df['distance_km'].min()
    n = len(qualifying_df)

    if n < MIN_RUNS_FOR_PERSONAL_EXPONENT or spread < MIN_DISTANCE_SPREAD_RATIO:
        msg = f"Using generic Riegel exponent ({RIEGEL_GENERIC_EXPONENT}) — not enough distance variety yet ({n} runs, {spread:.1f}x spread) to fit your own."
        return RIEGEL_GENERIC_EXPONENT, msg

    x = np.log(qualifying_df['distance_km'].values)
    y = np.log(qualifying_df['moving_time_sec'].values)
    w = recency_weights(qualifying_df['date'], qualifying_df['date'].max())
    k, _ = np.polyfit(x, y, deg=1, w=w)
    k = float(np.clip(k, *PERSONAL_EXPONENT_BOUNDS))
    msg = f"Fitted personal Riegel exponent: {k:.3f} (generic default is {RIEGEL_GENERIC_EXPONENT}) from {n} runs spanning {spread:.1f}x distance range."
    return k, msg

def riegel_convert_pace(pace_sec_km, from_km, to_km, exponent):
    return pace_sec_km * (to_km / from_km) ** (exponent - 1)

# ==========================================
# 3. WEEKLY AGGREGATION
# ==========================================
def aggregate_weekly(df, exponent, hr_available, effort_available):
    df = df.copy()
    df['pace_5k_equiv'] = riegel_convert_pace(df['pace_sec_km'], df['distance_km'], 5.0, exponent)

    qualifying = df[df['distance_km'] >= MIN_DISTANCE_KM].set_index('date')
    weekly = qualifying.resample('W').agg(
        weekly_km=('distance_km', 'sum'),
        long_run_km=('distance_km', 'max'),
        avg_temp=('temp', 'mean'),
        frac_weather_temp=('temp_reliable', 'mean'),
        weekly_elevation=('elevation_m', 'sum'),
        avg_hr=('hr', 'mean'),
        avg_training_load=('training_load', 'mean'),
        fitness_pace=('pace_5k_equiv', 'min'),
    ).dropna(subset=['weekly_km', 'fitness_pace'])

    if len(weekly) == 0:
        return weekly, False, False, 0, 0, 0
        
    weekly['avg_temp'] = weekly['avg_temp'].fillna(weekly['avg_temp'].mean())

    true_fitness_pace = weekly['fitness_pace'].min()
    threshold = true_fitness_pace * QUALITY_WEEK_PACE_MULTIPLIER
    before = len(weekly)
    weekly = weekly[weekly['fitness_pace'] <= threshold].copy()

    hr_signal = hr_available and weekly['avg_hr'].std() >= 2.0
    effort_signal = effort_available and weekly['avg_training_load'].notna().sum() >= 4
    return weekly, hr_signal, effort_signal, true_fitness_pace, before, len(weekly)

# ==========================================
# 4. MODELING & VALIDATION
# ==========================================
def build_features(weekly, hr_signal, effort_signal):
    features = ['weekly_km', 'long_run_km', 'avg_temp', 'weekly_elevation']
    if hr_signal:
        weekly['intensity_index'] = weekly['weekly_km'] * (weekly['avg_hr'] / 160)
        features += ['avg_hr', 'intensity_index']
    elif effort_signal:
        weekly['avg_training_load'] = weekly['avg_training_load'].fillna(weekly['avg_training_load'].mean())
        features += ['avg_training_load']
    return features

def sample_weights_for(weekly, reference_date):
    recency = recency_weights(weekly.index, reference_date)
    reliability = TEMP_RELIABILITY_FLOOR + (1 - TEMP_RELIABILITY_FLOOR) * weekly['frac_weather_temp'].fillna(0).values
    return recency * reliability

def train_quantile_models(X, y, weight):
    models = {}
    for name, q in QUANTILES.items():
        model = XGBRegressor(
            objective='reg:quantileerror', quantile_alpha=q,
            n_estimators=75 if len(X) < MIN_ROWS_FOR_TUNING else 150,
            max_depth=2, learning_rate=0.05, reg_alpha=0.5, reg_lambda=2.0,
            subsample=0.8, random_state=42, verbosity=0
        )
        model.fit(X, y, sample_weight=weight)
        models[name] = model
    imp = pd.Series(models['median'].feature_importances_, index=X.columns).sort_values(ascending=False)
    return models, imp

def walk_forward_validation(X, y, weight, dates, min_train=8):
    preds, actuals, pred_dates = [], [], []
    for i in range(min_train, len(X)):
        model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5,
                              n_estimators=75, max_depth=2, learning_rate=0.05,
                              reg_alpha=0.5, reg_lambda=2.0, subsample=0.8,
                              random_state=42, verbosity=0)
        model.fit(X.iloc[:i], y.iloc[:i], sample_weight=weight[:i])
        preds.append(model.predict(X.iloc[[i]])[0])
        actuals.append(y.iloc[i])
        pred_dates.append(dates[i])
    if not preds:
        return None
    errors = np.array(preds) - np.array(actuals)
    rmse = np.sqrt((errors ** 2).mean())
    return {'dates': pred_dates, 'predicted': preds, 'actual': actuals, 'rmse': rmse, 'bias': errors.mean()}

# ==========================================
# 5. OPTIMIZATION
# ==========================================
def build_bounds_and_density(weekly, cols, margin=EXTRAPOLATION_MARGIN):
    bounds, means, spans = [], [], []
    for c in cols:
        lo, hi = weekly[c].min(), weekly[c].max()
        span = max(hi - lo, 1e-6)
        lo, hi = max(lo - span * margin, 0 if c != 'avg_temp' else -50), hi + span * margin
        bounds.append((lo, hi))
    X = weekly[cols].values.astype(float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    return bounds, mean, std, (X - mean) / std

def density_penalty(point, mean, std, Xz, k=3):
    dists = np.linalg.norm(Xz - (np.array(point) - mean) / std, axis=1)
    return float(np.mean(np.sort(dists)[:min(k, len(dists))]))

def optimize_training(models, weekly, hr_signal, effort_signal, exponent):
    ga_cols = ['weekly_km', 'avg_temp', 'weekly_elevation']
    bounds, mean, std, Xz = build_bounds_and_density(weekly, ga_cols)
    fixed_hr = weekly['avg_hr'].mean()
    fixed_load = weekly['avg_training_load'].mean() if effort_signal else None

    def feature_row(v):
        weekly_km, avg_temp, weekly_elev = v
        row = {'weekly_km': weekly_km, 'long_run_km': weekly_km * 0.35,
               'avg_temp': avg_temp, 'weekly_elevation': weekly_elev}
        if hr_signal:
            row['avg_hr'] = fixed_hr
            row['intensity_index'] = weekly_km * (fixed_hr / 160)
        elif effort_signal:
            row['avg_training_load'] = fixed_load
        return row

    def objective(v):
        row = feature_row(v)
        X = pd.DataFrame([row])[models['median'].feature_names_in_]
        pred = float(models['median'].predict(X)[0])
        return pred + DENSITY_PENALTY_WEIGHT * density_penalty(v, mean, std, Xz)

    result = differential_evolution(objective, bounds, seed=42, popsize=15, mutation=(0.5, 1.5))
    v = result.x
    row = feature_row(v)
    predictions = {}
    for name, model in models.items():
        X = pd.DataFrame([row])[model.feature_names_in_]
        fitness_pace = float(model.predict(X)[0])
        predictions[name] = {
            label: riegel_convert_pace(fitness_pace, 5.0, dist_km, exponent) * dist_km
            for label, dist_km in TARGET_DISTANCES_KM.items()
        }
    return v, predictions

# ==========================================
# 6. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="PB Predictor", page_icon="🏃", layout="wide")

st.title("🏃 PB Predictor — Redesigned")
st.write("Upload your `activities.csv` from Strava/Garmin to calculate personalized 5K and 10K equivalent paces using Riegel's formula, alongside optimally tuned training targets.")

uploaded_file = st.file_uploader("Upload your activities data (CSV)", type="csv")

if uploaded_file is not None:
    with st.spinner("Processing data..."):
        df_raw = pd.read_csv(uploaded_file)
        df, hr_available, effort_available, report = load_and_clean(df_raw)
        qualifying = df[df['distance_km'] >= MIN_DISTANCE_KM]

        # --- DATA QUALITY REPORT ---
        st.subheader("📊 Data Quality Report")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Qualifying Runs", f"{report['total_runs']}")
        col2.metric("Distance Range", f"{report['min_dist']:.1f} - {report['max_dist']:.1f} km")
        col3.metric("Training Blocks", f"{report['blocks']}")
        col4.metric("Heart Rate Runs", f"{report['hr_runs']}" if hr_available else "Not Available")
        
        st.info(f"**Date Range:** {report['start_date']} to {report['end_date']} | **Temperature Data:** {report['weather_runs']} weather station, {report['device_runs']} device-only")

        if len(qualifying) < 4:
            st.error("Not enough qualifying runs to build a model (minimum 4).")
            st.stop()

        exponent, exp_msg = fit_personal_riegel_exponent(qualifying)
        st.write(f"**Riegel Exponent:** {exp_msg}")

        weekly, hr_signal, effort_signal, true_fitness_pace, before, after = aggregate_weekly(
            df, exponent, hr_available, effort_available
        )
        st.write(f"**Quality-week filter:** kept {after}/{before} weeks (fastest-effort weeks only, within {QUALITY_WEEK_PACE_MULTIPLIER}x of your best).")

        if len(weekly) < 4:
            st.error("Not enough quality weeks after filtering to train a model.")
            st.stop()

        features = build_features(weekly, hr_signal, effort_signal)
        X, y = weekly[features], weekly['fitness_pace']
        weight = sample_weights_for(weekly, weekly.index.max())

        models, imp = train_quantile_models(X, y, weight)
        wf_result = walk_forward_validation(X, y, weight, weekly.index)

        v, predictions = optimize_training(models, weekly, hr_signal, effort_signal, exponent)

        st.divider()

        # --- OPTIMAL TRAINING & PREDICTIONS ---
        st.subheader("🏆 Optimal Training & PB Predictions")
        
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown("### 🏃 Predictions")
            for label in TARGET_DISTANCES_KM:
                lo = predictions['low'][label]
                med = predictions['median'][label]
                hi = predictions['high'][label]
                st.metric(label=f"{label} Estimate", value=format_time(med), delta=f"Range: {format_time(hi)} – {format_time(lo)}", delta_color="off")
        
        with pcol2:
            st.markdown("### 🎯 Ideal Targets")
            st.metric("Weekly Volume", f"{v[0]:.1f} km")
            st.metric("Ideal Race Temp", f"{v[1]:.1f} °C")
            st.metric("Weekly Climb", f"{v[2]:.0f} m")
            if wf_result:
                st.metric("Walk-Forward Validation RMSE", f"{wf_result['rmse']:.1f} sec/km")

        # --- DASHBOARD (MATPLOTLIB) ---
        st.subheader("📈 Training & Race Dashboard")
        
        # Render the exact matplotlib logic provided in the prompt to st.pyplot()
        plt.rcParams.update({
            'font.family': 'sans-serif', 'font.size': 10,
            'axes.spines.top': False, 'axes.spines.right': False,
            'axes.edgecolor': '#888', 'axes.labelcolor': '#333',
            'text.color': '#222', 'axes.grid': True,
            'grid.alpha': 0.25, 'grid.linestyle': '--',
        })
        ACCENT = '#d9534f'   
        MAIN = '#2f6690'     
        MUTED = '#aab7c4'    

        fig = plt.figure(figsize=(13, 7))
        gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.25)

        # Middle-left: weekly volume over time
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(weekly.index, weekly['weekly_km'], width=5, color=MAIN, alpha=0.85)
        ax1.axhline(v[0], color=ACCENT, linestyle='--', linewidth=1.5, label=f"Optimal target ({v[0]:.0f} km)")
        ax1.set_title("Weekly Training Volume", fontsize=11, fontweight='bold')
        ax1.set_ylabel("km")
        ax1.legend(fontsize=8, frameon=False)
        ax1.tick_params(axis='x', rotation=30)

        # Middle-right: fitness pace trend
        ax2 = fig.add_subplot(gs[0, 1])
        pace_min = weekly['fitness_pace'] / 60
        ax2.plot(weekly.index, pace_min, 'o-', color=MAIN, markersize=4, linewidth=1.5, label="Weekly best (5K-equiv)")
        if len(weekly) >= 3:
            trend = weekly['fitness_pace'].rolling(3, min_periods=1).mean() / 60
            ax2.plot(weekly.index, trend, color=ACCENT, linewidth=2, label="3-week trend")
        ax2.invert_yaxis()
        ax2.set_title("Fitness Trend (5K-equivalent pace)", fontsize=11, fontweight='bold')
        ax2.set_ylabel("min/km")
        ax2.legend(fontsize=8, frameon=False)
        ax2.tick_params(axis='x', rotation=30)

        # Bottom-left: feature importance
        ax3 = fig.add_subplot(gs[1, 0])
        imp_sorted = imp.sort_values()
        colors = [ACCENT if i == len(imp_sorted) - 1 else MUTED for i in range(len(imp_sorted))]
        ax3.barh(imp_sorted.index, imp_sorted.values, color=colors)
        ax3.set_title("What Drives the Prediction", fontsize=11, fontweight='bold')
        ax3.set_xlabel("Relative importance")

        # Bottom-right: walk-forward validation
        ax4 = fig.add_subplot(gs[1, 1])
        if wf_result:
            actual_min = np.array(wf_result['actual']) / 60
            pred_min = np.array(wf_result['predicted']) / 60
            lims = [min(actual_min.min(), pred_min.min()) - 0.2, max(actual_min.max(), pred_min.max()) + 0.2]
            ax4.plot(lims, lims, '--', color=MUTED, linewidth=1, label="Perfect prediction")
            ax4.scatter(actual_min, pred_min, color=MAIN, s=45, alpha=0.8, edgecolor='white', linewidth=0.5)
            ax4.set_xlim(lims); ax4.set_ylim(lims)
            ax4.set_xlabel("Actual pace (min/km)")
            ax4.set_ylabel("Predicted pace (min/km)")
            ax4.legend(fontsize=8, frameon=False)
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, "Not enough data yet\nfor walk-forward validation", ha='center', va='center')
        ax4.set_title("Model Honesty Check\n(predicted vs. actual)", fontsize=11, fontweight='bold')

        st.pyplot(fig)
