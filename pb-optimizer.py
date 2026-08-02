import io
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scipy.optimize import differential_evolution
import warnings
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ CONFIG
# ==========================================
MIN_DISTANCE_KM = 4.8
EARLIEST_ACTIVITY_DATE = '2025-01-01'
QUALITY_WEEK_PACE_MULTIPLIER = 1.25
MIN_QUALITY_WEEKS_WARNING = 8  # warn if the quality filter leaves fewer weeks than this

RIEGEL_GENERIC_EXPONENT = 1.06
MIN_RUNS_FOR_PERSONAL_EXPONENT = 10
MIN_DISTANCE_SPREAD_RATIO = 1.5
PERSONAL_EXPONENT_BOUNDS = (1.00, 1.15)

RECENCY_HALF_LIFE_DAYS = 180
RECENCY_WEIGHT_FLOOR = 0.35
TEMP_RELIABILITY_FLOOR = 0.5

# Heuristic threshold for using more estimators — NOT a tuning/validation
# threshold, just "more data -> slightly bigger model." Named accordingly.
ROW_COUNT_FOR_LARGER_MODEL = 15
EXTRAPOLATION_MARGIN = 0.10
DENSITY_PENALTY_WEIGHT = 25.0
GA_MAXITER = 200  # capped explicitly so this can't hang the app indefinitely

TARGET_DISTANCES_KM = {'5K': 5.0, '10K': 10.0}
QUANTILES = {'low': 0.2, 'median': 0.5, 'high': 0.8}

REQUIRED_COLUMNS = {'Activity Date', 'Distance'}


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

    # FIXED: row-level fallback instead of column-level. The old version picked
    # ONE column for the entire dataframe (`Moving Time` if it exists at all,
    # else `Elapsed Time`), which silently drops rows that have one but not the
    # other. This backfills missing Moving Time values from Elapsed Time
    # per-row instead.
    moving = safe_numeric(df, 'Moving Time') if 'Moving Time' in df.columns else pd.Series(np.nan, index=df.index)
    elapsed = safe_numeric(df, 'Elapsed Time') if 'Elapsed Time' in df.columns else pd.Series(np.nan, index=df.index)
    # Moving Time is sometimes formatted as "H:MM:SS" text rather than raw
    # seconds, so parse both through safe_parse_time on the original columns.
    moving_parsed = (df['Moving Time'].apply(safe_parse_time) if 'Moving Time' in df.columns
                      else pd.Series(np.nan, index=df.index))
    elapsed_parsed = (df['Elapsed Time'].apply(safe_parse_time) if 'Elapsed Time' in df.columns
                       else pd.Series(np.nan, index=df.index))
    df['moving_time_sec'] = moving_parsed.fillna(elapsed_parsed)

    dist_raw = safe_numeric(df, 'Distance')
    df['distance_km'] = dist_raw / 1000.0 if dist_raw.max() > 500 else dist_raw

    # FIXED: guard against zero/negative distance (GPS glitches, bad manual
    # entries) producing inf/negative pace that dropna() wouldn't catch.
    df = df.dropna(subset=['distance_km', 'moving_time_sec'])
    df = df[df['distance_km'] > 0]

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
        'min_dist': qualifying['distance_km'].min() if len(qualifying) else None,
        'max_dist': qualifying['distance_km'].max() if len(qualifying) else None,
        'start_date': qualifying['date'].min().date() if len(qualifying) else None,
        'end_date': qualifying['date'].max().date() if len(qualifying) else None,
        'blocks': n_blocks,
        'hr_runs': int(df['hr'].notna().sum()),
        'load_runs': int(df['training_load'].notna().sum()),
        'weather_runs': int(weather_temp.notna().sum()),
        'device_runs': int((weather_temp.isna() & device_temp.notna()).sum()),
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
        msg = (f"Using generic Riegel exponent ({RIEGEL_GENERIC_EXPONENT}) — not enough distance "
               f"variety yet ({n} runs, {spread:.1f}x spread) to fit your own.")
        return RIEGEL_GENERIC_EXPONENT, msg

    x = np.log(qualifying_df['distance_km'].values)
    y = np.log(qualifying_df['moving_time_sec'].values)
    w = recency_weights(qualifying_df['date'], qualifying_df['date'].max())
    k, _ = np.polyfit(x, y, deg=1, w=w)
    k = float(np.clip(k, *PERSONAL_EXPONENT_BOUNDS))
    msg = (f"Fitted personal Riegel exponent: {k:.3f} (generic default is {RIEGEL_GENERIC_EXPONENT}) "
           f"from {n} runs spanning {spread:.1f}x distance range.")
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
            n_estimators=75 if len(X) < ROW_COUNT_FOR_LARGER_MODEL else 150,
            max_depth=2, learning_rate=0.05, reg_alpha=0.5, reg_lambda=2.0,
            subsample=0.8, random_state=42, verbosity=0
        )
        model.fit(X, y, sample_weight=weight)
        models[name] = model
    imp = pd.Series(models['median'].feature_importances_, index=X.columns).sort_values(ascending=False)
    return models, imp


def predict_quantiles_sorted(models, X_row):
    """
    FIXED: quantile crossing. Each quantile is trained as an independent
    model, so nothing guarantees low <= median <= high at any given input,
    especially with this little data. Sort the three outputs post-hoc so the
    displayed range is always coherent, rather than occasionally showing a
    "low" estimate slower than the "median" one.
    """
    raw = {name: float(model.predict(X_row)[0]) for name, model in models.items()}
    ordered_values = sorted(raw.values())
    names_by_quantile = sorted(QUANTILES, key=lambda n: QUANTILES[n])
    return dict(zip(names_by_quantile, ordered_values))


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
    bounds = []
    for c in cols:
        lo, hi = weekly[c].min(), weekly[c].max()
        span = max(hi - lo, 1e-6)
        lo = max(lo - span * margin, 0)
        hi = hi + span * margin
        bounds.append((lo, hi))
    X = weekly[cols].values.astype(float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    return bounds, mean, std, (X - mean) / std


def density_penalty(point, mean, std, Xz, k=3):
    dists = np.linalg.norm(Xz - (np.array(point) - mean) / std, axis=1)
    return float(np.mean(np.sort(dists)[:min(k, len(dists))]))


def optimize_training(models, weekly, hr_signal, effort_signal, exponent):
    """
    FIXED (conceptual bug): temperature used to be a GA decision variable,
    which frames it as something a runner can choose — it can't be. Race-day
    temperature is a condition you observe, not a training input you control.

    Now only weekly_km and weekly_elevation are optimized. avg_temp is held
    fixed at the median temperature across your quality weeks, and reported
    separately as an *observation* ("your best efforts tend to happen around
    X°C") rather than folded into the "optimal training targets" the GA
    actually searches over.

    Also FIXED: long_run_km used to be hardcoded as weekly_km * 0.35. That's
    now derived from your own data — the median long_run_km / weekly_km ratio
    across your quality weeks — so the recommendation reflects how you
    actually structure your own long runs, not a generic assumption.
    """
    ga_cols = ['weekly_km', 'weekly_elevation']
    bounds, mean, std, Xz = build_bounds_and_density(weekly, ga_cols)

    reference_temp = float(weekly['avg_temp'].median())
    fixed_hr = weekly['avg_hr'].mean()
    fixed_load = weekly['avg_training_load'].mean() if effort_signal else None

    ratios = (weekly['long_run_km'] / weekly['weekly_km']).replace([np.inf, -np.inf], np.nan).dropna()
    long_run_ratio = float(ratios.median()) if len(ratios) else 0.35

    def feature_row(v):
        weekly_km, weekly_elev = v
        row = {'weekly_km': weekly_km, 'long_run_km': weekly_km * long_run_ratio,
               'avg_temp': reference_temp, 'weekly_elevation': weekly_elev}
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

    result = differential_evolution(objective, bounds, seed=42, popsize=15,
                                     mutation=(0.5, 1.5), maxiter=GA_MAXITER)
    v = result.x
    row = feature_row(v)

    predictions = {}
    for label, dist_km in TARGET_DISTANCES_KM.items():
        fitness_quantiles = predict_quantiles_sorted(
            models, pd.DataFrame([row])[models['median'].feature_names_in_]
        )
        for name, fitness_pace in fitness_quantiles.items():
            predictions.setdefault(name, {})[label] = riegel_convert_pace(fitness_pace, 5.0, dist_km, exponent) * dist_km

    return v, predictions, reference_temp, long_run_ratio


# ==========================================
# 6. CACHED PIPELINE
# ==========================================
@st.cache_data(show_spinner=False)
def process_pipeline(file_bytes):
    df_raw = pd.read_csv(io.BytesIO(file_bytes))
    df, hr_available, effort_available, report = load_and_clean(df_raw)
    qualifying = df[df['distance_km'] >= MIN_DISTANCE_KM]

    if len(qualifying) < 4:
        return {'error': 'Not enough qualifying runs to build a model (minimum 4).'}

    exponent, exp_msg = fit_personal_riegel_exponent(qualifying)
    weekly, hr_signal, effort_signal, true_fitness_pace, before, after = aggregate_weekly(
        df, exponent, hr_available, effort_available
    )
    if len(weekly) < 4:
        return {'error': 'Not enough quality weeks after filtering to train a model.'}

    features = build_features(weekly, hr_signal, effort_signal)
    X, y = weekly[features], weekly['fitness_pace']
    weight = sample_weights_for(weekly, weekly.index.max())

    models, imp = train_quantile_models(X, y, weight)
    wf_result = walk_forward_validation(X, y, weight, weekly.index)
    v, predictions, reference_temp, long_run_ratio = optimize_training(
        models, weekly, hr_signal, effort_signal, exponent
    )

    return {
        'error': None, 'report': report, 'exponent': exponent, 'exp_msg': exp_msg,
        'weekly': weekly, 'before': before, 'after': after, 'models': models, 'imp': imp,
        'wf_result': wf_result, 'v': v, 'predictions': predictions,
        'reference_temp': reference_temp, 'long_run_ratio': long_run_ratio,
    }


# ==========================================
# 7. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="PB Predictor", page_icon="🏃", layout="wide")
st.title("🏃 PB Predictor — Redesigned")
st.write("Upload your `activities.csv` from Strava/Garmin to calculate personalized 5K and 10K "
         "equivalent paces using Riegel's formula, alongside optimally tuned training targets.")

uploaded_file = st.file_uploader("Upload your activities data (CSV)", type="csv")

if uploaded_file is not None:
    try:
        preview = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), nrows=1)
    except Exception:
        st.error("Couldn't read that file as a CSV. Please check the export and try again.")
        st.stop()

    missing = REQUIRED_COLUMNS - set(preview.columns.str.strip())
    if missing:
        st.error(f"This doesn't look like a Strava/Garmin export — missing columns: {', '.join(missing)}")
        st.stop()

    with st.spinner("Processing your data..."):
        results = process_pipeline(uploaded_file.getvalue())

    if results['error']:
        st.error(results['error'])
        st.stop()

    report = results['report']
    weekly = results['weekly']
    v, predictions = results['v'], results['predictions']
    wf_result = results['wf_result']

    # --- DATA QUALITY REPORT ---
    st.subheader("📊 Data Quality Report")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Qualifying Runs", f"{report['total_runs']}")
    col2.metric("Distance Range", f"{report['min_dist']:.1f}–{report['max_dist']:.1f} km")
    col3.metric("Training Blocks", f"{report['blocks']}")
    col4.metric("Heart Rate Runs", f"{report['hr_runs']}" if report['hr_runs'] >= 4 else "Not available")

    st.info(f"**Date Range:** {report['start_date']} to {report['end_date']}  |  "
            f"**Temperature Data:** {report['weather_runs']} weather station, {report['device_runs']} device-only")

    if report['blocks'] >= 4:
        st.warning(f"⚠️ Your training has {report['blocks']} separate blocks with 21+ day gaps between them — "
                   "predictions reflect a fragmented history, not continuous training.")

    st.write(f"**Riegel Exponent:** {results['exp_msg']}")
    st.write(f"**Quality-week filter:** kept {results['after']}/{results['before']} weeks "
             f"(fastest-effort weeks only, within {QUALITY_WEEK_PACE_MULTIPLIER}x of your best).")
    if results['after'] < MIN_QUALITY_WEEKS_WARNING:
        st.warning(f"⚠️ Only {results['after']} quality weeks after filtering — below "
                   f"{MIN_QUALITY_WEEKS_WARNING}, predictions here should be treated as rough estimates.")

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
            st.metric(label=f"{label} Estimate", value=format_time(med),
                      delta=f"Range: {format_time(lo)}–{format_time(hi)}", delta_color="off")

    with pcol2:
        st.markdown("### 🎯 Controllable Training Targets")
        st.metric("Weekly Volume", f"{v[0]:.1f} km")
        st.metric("Weekly Climb", f"{v[1]:.0f} m")
        if wf_result:
            st.metric("Walk-Forward Validation RMSE", f"{wf_result['rmse']:.1f} sec/km")
        st.caption(f"Race temperature isn't something you can control, so it's not optimized above. "
                   f"Your quality weeks tend to cluster around **{results['reference_temp']:.1f}°C** — "
                   f"that's an observation, not a target to chase.")

    if st.button("⬇️ Prepare results for download"):
        out_df = pd.DataFrame([{
            'Target_Weekly_Volume_km': round(v[0], 1),
            'Target_Weekly_Elevation_m': round(v[1], 0),
            'Reference_Temperature_C': round(results['reference_temp'], 1),
            '5K_median': format_time(predictions['median']['5K']),
            '5K_range': f"{format_time(predictions['low']['5K'])}-{format_time(predictions['high']['5K'])}",
            '10K_median': format_time(predictions['median']['10K']),
            '10K_range': f"{format_time(predictions['low']['10K'])}-{format_time(predictions['high']['10K'])}",
        }])
        st.download_button("Download predictions CSV", out_df.to_csv(index=False),
                            file_name="pb_predictions.csv", mime="text/csv")

    # --- INTERACTIVE DASHBOARD (PLOTLY) ---
    st.subheader("📈 Training & Race Dashboard")
    tab1, tab2, tab3, tab4 = st.tabs(["Weekly Volume", "Fitness Trend", "What Drives It", "Model Accuracy"])

    with tab1:
        fig = px.bar(weekly, x=weekly.index, y='weekly_km', title="Weekly Training Volume",
                     labels={'weekly_km': 'km', 'x': 'Week'})
        fig.add_hline(y=v[0], line_dash="dash", line_color="#d9534f",
                      annotation_text=f"Optimal target ({v[0]:.0f} km)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        pace_min = weekly['fitness_pace'] / 60
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekly.index, y=pace_min, mode='lines+markers', name='Weekly best (5K-equiv)'))
        if len(weekly) >= 3:
            trend = weekly['fitness_pace'].rolling(3, min_periods=1).mean() / 60
            fig.add_trace(go.Scatter(x=weekly.index, y=trend, mode='lines', name='3-week trend',
                                      line=dict(color="#d9534f", width=3)))
        fig.update_yaxes(autorange="reversed", title="min/km (lower = faster)")
        fig.update_layout(title="Fitness Trend (5K-equivalent pace)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        imp_sorted = results['imp'].sort_values()
        fig = px.bar(x=imp_sorted.values, y=imp_sorted.index, orientation='h',
                     title="What Drives the Prediction", labels={'x': 'Relative importance', 'y': ''})
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if wf_result:
            actual_min = np.array(wf_result['actual']) / 60
            pred_min = np.array(wf_result['predicted']) / 60
            lims = [min(actual_min.min(), pred_min.min()) - 0.2, max(actual_min.max(), pred_min.max()) + 0.2]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lims, y=lims, mode='lines', name='Perfect prediction',
                                      line=dict(dash='dash', color='gray')))
            fig.add_trace(go.Scatter(x=actual_min, y=pred_min, mode='markers', name='Predicted vs actual',
                                      marker=dict(size=10)))
            fig.update_layout(title="Model Honesty Check (out-of-sample)",
                               xaxis_title="Actual pace (min/km)", yaxis_title="Predicted pace (min/km)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data yet for walk-forward validation.")
