import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import json
import os
import zipfile

warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import boxcox, jarque_bera, kurtosis as sp_kurtosis
from scipy.special import inv_boxcox
from sklearn.metrics import classification_report, confusion_matrix
from pmdarima import auto_arima

# =============================================================================
# PATHS
# =============================================================================

DATA_FILE  = 'vru_data_full_4years.csv'   # ganti path kalau perlu
FALLBACK_DATA_FILE = 'vru_preprocessed.csv'
OUT_ROOT   = 'output/'
DIR_MODELS = OUT_ROOT + 'arima_models/'
DIR_ALERTS = OUT_ROOT + 'alerts/'
DIR_PLOTS  = OUT_ROOT + 'plots/'
DIR_PERF   = OUT_ROOT + 'performance_results/'

for d in [DIR_MODELS, DIR_ALERTS, DIR_PLOTS, DIR_PERF]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# CONFIG
# =============================================================================

PARAMS       = ['discharge_temp', 'discharge_pressure', 'jacket_water']
PARAM_UNITS  = {'discharge_temp': '°F', 'discharge_pressure': 'psi', 'jacket_water': 'psi'}
PARAM_LABELS = {
    'discharge_temp':     'Discharge Temperature',
    'discharge_pressure': 'Discharge Pressure',
    'jacket_water':       'Jacket Water Pressure',
}

TRAIN_RATIO    = 0.80   # FIX: 80/20 split (dari 85/15)
HORIZON        = 5
REFIT_INTERVAL = 20

CUSUM_K_FACTOR = 0.5
CUSUM_H_FACTOR = 4.0   # restored: EWMA yang memberikan early warning (Lucas & Saccucci 1990)

PI_WINDOW      = 7
PI_LEVEL       = 0.95
PI_YELLOW_RATE = 2 / 7
PI_RED_RATE    = 4 / 7

EWMA_LAMBDA    = 0.3   # FIX H3: EWMA smoothing factor — Lucas & Saccucci (1990) Technometrics 32
EWMA_L         = 2.5   # FIX H3: EWMA control limit multiplier (lebih konservatif dari CUSUM)

THRESHOLDS = {
    'discharge_temp':     {'OFF': 90, 'YELLOW': 150, 'RED': 160},
    'discharge_pressure': {'LOW': 10, 'HIGH': 30},
    'jacket_water':       {'LOW': 12, 'HIGH': 20},
}

PI_THRESHOLDS = {
    'discharge_temp':     {'yellow': 2/7, 'red': 4/7},
    'discharge_pressure': {'yellow': 3/7, 'red': 5/7},
    'jacket_water':       {'yellow': 4/7, 'red': 6/7},
}

CONFIRM = 1   # FIX: dari 2 ke 1

TARGETS = {
    'MAPE_1day':    10.0,
    'MAPE_5day':    15.0,
    'MASE':          1.0,
    'Lead_Time_Avg': 3.0,
}

LEVEL = {'GREEN': 0, 'YELLOW': 1, 'RED': 2, 'OFF': -1}
REV   = {0: 'GREEN', 1: 'YELLOW', 2: 'RED', -1: 'OFF'}

print('Config loaded')
print(f'  Data file  : {DATA_FILE}')
print(f'  Split      : {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}')
print(f'  Horizon    : {HORIZON} days')
print(f'  CONFIRM    : {CONFIRM}')


if not os.path.exists(DATA_FILE) and os.path.exists(FALLBACK_DATA_FILE):
    print(f'\nData file not found: {DATA_FILE} — using {FALLBACK_DATA_FILE}')
    DATA_FILE = FALLBACK_DATA_FILE

df_full = pd.read_csv(DATA_FILE, parse_dates=['date'])
df_full = df_full.sort_values('date').reset_index(drop=True)

print(f'\nFull dataset : {len(df_full)} rows  ({df_full.date.min().date()} to {df_full.date.max().date()})')

# Remove machine-down periods
if 'status' in df_full.columns:
    df = df_full[df_full['status'] != 'down'].copy().reset_index(drop=True)
    print(f'After removing DOWN: {len(df)} rows')
else:
    df = df_full.copy().reset_index(drop=True)
    print('Status column not found; assuming down-days were already removed in preprocessing')

# Reindex to daily calendar, forward-fill short gaps
df = df.set_index('date')
full_idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
df = df.reindex(full_idx)
for col in PARAMS:
    df[col] = df[col].ffill(limit=3)
df = df.reset_index().rename(columns={'index': 'date'})

df_valid = df.dropna(subset=PARAMS).reset_index(drop=True)
print(f'Valid rows   : {len(df_valid)}')

# Train/test split — FIX: 80/20
split_idx = int(len(df_valid) * TRAIN_RATIO)
train = df_valid.iloc[:split_idx].copy()
test  = df_valid.iloc[split_idx:].copy().reset_index(drop=True)

print(f'Train        : {len(train)} rows  ({train.date.min().date()} to {train.date.max().date()})')
print(f'Test         : {len(test)} rows   ({test.date.min().date()} to {test.date.max().date()})')

# =============================================================================
# PLOT: TIME SERIES OVERVIEW
# =============================================================================

colors = {'running': '#3B82F6', 'degrading': '#F59E0B', 'down': '#EF4444'}
fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

for ax, param in zip(axes, PARAMS):
    if 'status' in df_full.columns:
        for status, grp in df_full.groupby('status'):
            ax.scatter(grp['date'], grp[param], s=3, alpha=0.6,
                       color=colors.get(status, 'gray'), label=status)
    else:
        ax.scatter(df_full['date'], df_full[param], s=3, alpha=0.6,
                   color=colors.get('running', '#3B82F6'), label='preprocessed')
    ax.axvline(train.date.max(), color='black', linestyle='--', linewidth=1,
               label='Train/Test split')
    ax.set_ylabel(f"{PARAM_LABELS[param]}\n({PARAM_UNITS[param]})", fontsize=9)
    ax.grid(True, alpha=0.3)

if 'status' in df_full.columns:
    handles = [mpatches.Patch(color=c, label=s) for s, c in colors.items()]
else:
    handles = [mpatches.Patch(color=colors.get('running', '#3B82F6'), label='preprocessed')]
handles.append(plt.Line2D([0], [0], color='black', linestyle='--', label='Train/Test split'))
axes[0].legend(handles=handles, loc='upper left', fontsize=8)
axes[0].set_title('VRU Compressor — Full Time Series (4 Years)', fontsize=13)
axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig(DIR_PLOTS + '01_preprocessed_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\nPlot saved: {DIR_PLOTS}01_preprocessed_timeseries.png')


def detect_sarima_order(series, param_name):
    """
    Auto SARIMA order detection using pmdarima.
    Mencari kombinasi (p,d,q) x (P,D,Q,12) terbaik berdasarkan AIC.
    """
    print(f"  Running auto_arima for {param_name} (this may take a minute)...")
    
    # auto_arima otomatis mencari d dan D via KPSS/ADF test
    # dan mencari p,q,P,Q via stepwise search meminimalkan AIC
    model = auto_arima(
        series.dropna(),
        seasonal=True,
        m=12,                 # seasonality (12 months approx)
        d=None, D=None,       # let auto_arima find optimal differencing
        max_p=3, max_q=3,
        max_P=2, max_Q=2,
        start_p=0, start_q=0,
        start_P=0, start_Q=0,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True         # fast search
    )
    
    order = model.order
    seasonal_order = model.seasonal_order
    
    print(f'  auto_arima order: SARIMA{order}x{seasonal_order}')
    return order, seasonal_order


# FIX 1: nama fungsi yang benar adalah fit_sarima_boxcox, bukan fit_boxcox
def fit_sarima_boxcox(series, order, seasonal_order):
    """
    Fit SARIMA setelah Box-Cox transformation.
    Box & Cox (1964) JRSS-B 26:211-252
    """
    shift = 0
    if series.min() <= 0:
        shift = abs(series.min()) + 1.0
    s_shifted = series + shift

    transformed, lam = boxcox(s_shifted.dropna())
    ts_transformed = pd.Series(transformed, index=series.dropna().index)

    model = SARIMAX(
        ts_transformed,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=200)
    return fit, lam, shift


print('\nSARIMA ORDER DETECTION')
print('─' * 60)
orders_all = {}
for param in PARAMS:
    print(f'\n  {PARAM_LABELS[param]}:')
    o, so = detect_sarima_order(train[param], param)
    orders_all[param] = {'order': o, 'seasonal_order': so}

# Naive MAE baseline (random walk)
naive_maes = {}
for param in PARAMS:
    series = df_valid[param].values
    naive_maes[param] = float(np.mean(np.abs(np.diff(series))))
    print(f'  Naive MAE [{param}]: {naive_maes[param]:.4f}')


print('\nStarting rolling walk-forward forecast...')
print(f'  Test points  : {len(test)}')
print(f'  Horizon      : {HORIZON} days')
print(f'  Refit every  : {REFIT_INTERVAL} steps')

forecast_results = {}
fallback_counts  = {}
n_test = len(test)

for param in PARAMS:
    print(f'\n--- {PARAM_LABELS[param]} ---')
    o  = orders_all[param]['order']
    so = orders_all[param]['seasonal_order']

    records       = []
    fit           = None
    raw_fit       = None
    lam_bc        = None
    shift_bc      = None
    history_vals  = list(train[param].values)
    fallback_n    = 0

    for i in range(n_test - HORIZON):
        fc_date = test.iloc[i]['date']

        if fit is None or i % REFIT_INTERVAL == 0:
            try:
                hist_s = pd.Series(history_vals, dtype=float)
                fit, lam_bc, shift_bc = fit_sarima_boxcox(hist_s, o, so)
                # Raw SARIMA kept as fallback for when inv_boxcox returns NaN
                raw_fit = SARIMAX(
                    hist_s.dropna(), order=o, seasonal_order=so,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False, maxiter=200)
                print(f'  [refit] step {i:3d}/{n_test - HORIZON}  ({fc_date.date()})  AIC={fit.aic:.1f}')
            except Exception as exc:
                print(f'  [refit FAILED step {i}]: {exc}')
                fit = None; raw_fit = None
                history_vals.append(float(test.iloc[i][param]))
                continue

        try:
            fc_obj     = fit.get_forecast(steps=HORIZON)
            fc_mean_bc = fc_obj.predicted_mean.values
            fc_ci_bc   = fc_obj.conf_int(alpha=1 - PI_LEVEL)

            fc_mean  = inv_boxcox(fc_mean_bc,                  lam_bc) - shift_bc
            ci_lower = inv_boxcox(fc_ci_bc.iloc[:, 0].values,  lam_bc) - shift_bc
            ci_upper = inv_boxcox(fc_ci_bc.iloc[:, 1].values,  lam_bc) - shift_bc

            # Two failure modes require fallback to raw SARIMA:
            # 1. inv_boxcox returns NaN silently when lam*x+1 <= 0.
            # 2. Box-Cox SARIMA extrapolates to physically impossible values
            #    (e.g. discharge_temp < OFF threshold while compressor is running).
            hist_arr = np.asarray(history_vals, dtype=float)
            hist_med = float(np.nanmedian(hist_arr[-min(len(hist_arr), 120):]))
            temp_off = THRESHOLDS['discharge_temp']['OFF']
            implausible = (
                not np.all(np.isfinite(fc_mean))
                or (param == 'discharge_temp'
                    and hist_med >= temp_off
                    and float(np.nanmedian(fc_mean)) < temp_off)
            )
            if implausible and raw_fit is not None:
                fc_obj_raw = raw_fit.get_forecast(steps=HORIZON)
                fc_mean    = fc_obj_raw.predicted_mean.values
                fc_ci_raw  = fc_obj_raw.conf_int(alpha=1 - PI_LEVEL)
                ci_lower   = fc_ci_raw.iloc[:, 0].values
                ci_upper   = fc_ci_raw.iloc[:, 1].values
                fallback_n += 1

            tgt_1d = test.iloc[i + 1]
            tgt_5d = test.iloc[i + HORIZON - 1]

            records.append({
                'forecast_date': fc_date,
                'target_date':   tgt_1d['date'],
                'target_date_1day': tgt_1d['date'],
                'target_date_5day': tgt_5d['date'],
                'actual_1day':   float(tgt_1d[param]),
                'forecast_1day': float(fc_mean[0]),
                'pi_lower_1day': float(ci_lower[0]),
                'pi_upper_1day': float(ci_upper[0]),
                'actual_5day':   float(tgt_5d[param]),
                'forecast_5day': float(fc_mean[HORIZON - 1]),
                'pi_lower_5day': float(ci_lower[HORIZON - 1]),
                'pi_upper_5day': float(ci_upper[HORIZON - 1]),
            })
        except Exception as exc:
            print(f'  [forecast FAILED step {i}]: {exc}')

        history_vals.append(float(test.iloc[i][param]))

    fallback_counts[param] = fallback_n
    if fallback_n:
        print(f'  Raw-scale SARIMA fallback used for {fallback_n} forecast windows')

    df_fc = pd.DataFrame(records)
    if not df_fc.empty:
        df_fc['forecast_date'] = pd.to_datetime(df_fc['forecast_date'])
        df_fc['target_date']   = pd.to_datetime(df_fc['target_date'])
        df_fc['target_date_1day'] = pd.to_datetime(df_fc['target_date_1day'])
        df_fc['target_date_5day'] = pd.to_datetime(df_fc['target_date_5day'])
        df_fc.to_csv(DIR_MODELS + f'forecast_{param}.csv', index=False)
        df_fc.to_csv(DIR_MODELS + f'forecasts_{param}.csv', index=False)
        print(f'  {len(df_fc)} records -> {DIR_MODELS}forecast_{param}.csv')

    forecast_results[param] = df_fc

print('\nForecast done:')
for param in PARAMS:
    print(f'  {param}: {len(forecast_results.get(param, []))} rows')

# Dedicated 5-day forecast chart. The alert timeline below combines forecasts
# with alerts; this one is only for the 5-day forecast-vs-actual result.
fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
for ax, param in zip(axes, PARAMS):
    df_fc = forecast_results[param]
    dates_5d = df_fc['target_date_5day'] if 'target_date_5day' in df_fc else df_fc['target_date']
    ax.plot(dates_5d, df_fc['actual_5day'], color='#1E40AF', linewidth=1.1, label='Actual at 5-day target')
    ax.plot(dates_5d, df_fc['forecast_5day'], color='#EF4444', linewidth=1.1, linestyle='--', label='SARIMA 5-day forecast')
    ax.fill_between(dates_5d, df_fc['pi_lower_5day'], df_fc['pi_upper_5day'],
                    alpha=0.14, color='#EF4444', label='95% PI (5-day)')
    ax.set_ylabel(f"{PARAM_LABELS[param]}\n({PARAM_UNITS[param]})", fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
fig.suptitle('S4 SARIMA + Box-Cox + CUSUM + EWMA - 5-Day Forecast vs Actual', fontsize=12)
plt.tight_layout()
plt.savefig(DIR_PLOTS + 'forecast_vs_actual_5day.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"5-day forecast plot saved: {DIR_PLOTS}forecast_vs_actual_5day.png")

# =============================================================================
# DIAGNOSTIC PLOTS
# =============================================================================

for param in PARAMS:
    if param not in forecast_results or forecast_results[param].empty:
        continue

    df_fc     = forecast_results[param]
    residuals = df_fc['actual_1day'] - df_fc['forecast_1day']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_acf(train[param].dropna(),  lags=40, ax=axes[0, 0], title='ACF — Training Data')
    plot_pacf(train[param].dropna(), lags=40, ax=axes[0, 1], title='PACF — Training Data', method='ywm')

    axes[1, 0].plot(df_fc['target_date'], residuals, color='#3B82F6', linewidth=0.8)
    axes[1, 0].axhline(0, color='black', linewidth=0.8)
    axes[1, 0].axhline( residuals.std() * 2, color='orange', linestyle='--', linewidth=1, label='±2σ')
    axes[1, 0].axhline(-residuals.std() * 2, color='orange', linestyle='--', linewidth=1)
    axes[1, 0].set_title('Forecast Residuals (1-day)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    jb_stat, jb_p = jarque_bera(residuals.dropna())
    kurt = float(sp_kurtosis(residuals.dropna()))
    axes[1, 1].hist(residuals, bins=30, color='#3B82F6', alpha=0.7, edgecolor='white')
    axes[1, 1].set_title(f'Residual Distribution\nJB p={jb_p:.4f}, Kurt={kurt:.2f}')
    axes[1, 1].set_xlabel(f'Residual ({PARAM_UNITS[param]})')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'SARIMA Diagnostics — {PARAM_LABELS[param]}', fontsize=13)
    plt.tight_layout()
    plt.savefig(DIR_PLOTS + f'diagnostics_{param}.png', dpi=150, bbox_inches='tight')
    plt.close()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, param in zip(axes, PARAMS):
    ax.hist(train[param].dropna(), bins=40, color='#3B82F6', alpha=0.7, edgecolor='white')
    ax.set_title(PARAM_LABELS[param])
    ax.set_xlabel(PARAM_UNITS[param])
    ax.grid(True, alpha=0.3)
fig.suptitle('Training Data Distributions', fontsize=12)
plt.tight_layout()
plt.savefig(DIR_PLOTS + '01_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print('Diagnostic plots saved')



def compute_cusum(residuals, k_factor=0.5, h_factor=4.0):
    """CUSUM on ARIMA residuals — Page (1954) Biometrika."""
    res    = np.array(residuals, dtype=float)
    mean_r = np.nanmean(res)
    std_r  = np.nanstd(res)
    k = k_factor * std_r
    h = h_factor * std_r
    cp    = np.zeros(len(res))
    cn    = np.zeros(len(res))
    alert = np.zeros(len(res), dtype=bool)
    for i in range(1, len(res)):
        if np.isnan(res[i]):
            cp[i] = cp[i-1]; cn[i] = cn[i-1]; continue
        cp[i] = max(0.0, cp[i-1] + (res[i] - mean_r) - k)
        cn[i] = max(0.0, cn[i-1] - (res[i] - mean_r) - k)
        if cp[i] > h or cn[i] > h:
            alert[i] = True
            cp[i] = 0.0; cn[i] = 0.0
    return pd.DataFrame({
        'cusum_pos':   cp, 'cusum_neg': cn, 'cusum_alert': alert,
        'cusum_h':     h,  'residual':  res,
        'residual_mean': mean_r, 'residual_std': std_r,
    })


def compute_ewma(residuals, lam=0.3, L=2.5):
    """
    FIX H3: EWMA control chart pada ARIMA residuals.
    Lebih responsif terhadap drift kecil yang persisten dibanding CUSUM.
    Memberikan early warning lebih awal (lead time > 3 hari).
    Lucas & Saccucci (1990) Technometrics 32:778-792
    """
    res    = np.array(residuals, dtype=float)
    mean_r = np.nanmean(res)
    std_r  = np.nanstd(res)
    # UCL/LCL steady-state
    sigma_ewma = std_r * np.sqrt(lam / (2 - lam))
    ucl = mean_r + L * sigma_ewma
    lcl = mean_r - L * sigma_ewma
    z     = np.zeros(len(res))
    alert = np.zeros(len(res), dtype=bool)
    z[0]  = mean_r
    for i in range(1, len(res)):
        if np.isnan(res[i]):
            z[i] = z[i-1]; continue
        z[i] = lam * res[i] + (1 - lam) * z[i-1]
        if z[i] > ucl or z[i] < lcl:
            alert[i] = True
    return pd.DataFrame({
        'ewma_z': z, 'ewma_alert': alert,
        'ewma_ucl': ucl, 'ewma_lcl': lcl,
    })


def compute_pi_violation_rate(actuals, pi_uppers, pi_lowers, window=7):
    """Rolling PI violation rate — Chatfield (1993) JBES."""
    actuals   = np.array(actuals,   dtype=float)
    pi_uppers = np.array(pi_uppers, dtype=float)
    pi_lowers = np.array(pi_lowers, dtype=float)
    violated  = (actuals > pi_uppers) | (actuals < pi_lowers)
    rate      = np.full(len(actuals), np.nan)
    for i in range(window - 1, len(actuals)):
        rate[i] = violated[i - window + 1 : i + 1].mean()
    return rate, violated


def classify_param_status(value, param):
    thr = THRESHOLDS.get(param, {})
    if param == 'discharge_temp':
        if value < thr.get('OFF', 90):      return 'OFF'
        if value >= thr.get('RED', 160):    return 'RED'
        if value >= thr.get('YELLOW', 150): return 'YELLOW'
        return 'GREEN'
    elif param in ('discharge_pressure', 'jacket_water'):
        lo = thr.get('LOW', 0); hi = thr.get('HIGH', 9999)
        if value < lo or value > hi: return 'YELLOW'
        return 'GREEN'
    return 'GREEN'

base = forecast_results[PARAMS[0]][['forecast_date', 'target_date']].copy()
cusum_results = {}
ewma_results  = {}
pi_results    = {}

for param in PARAMS:
    df_fc = forecast_results[param]
    if df_fc.empty:
        continue
    residuals = df_fc['actual_1day'].values - df_fc['forecast_1day'].values
    cusum_results[param] = compute_cusum(residuals, CUSUM_K_FACTOR, CUSUM_H_FACTOR)
    ewma_results[param]  = compute_ewma(residuals, EWMA_LAMBDA, EWMA_L)
    rate, violated = compute_pi_violation_rate(
        df_fc['actual_1day'].values,
        df_fc['pi_upper_1day'].values,
        df_fc['pi_lower_1day'].values,
        window=PI_WINDOW,
    )
    pi_results[param] = {'rate': rate, 'violated': violated}

# BUILD ALERT TABLE — FIX 2 (voting) + FIX 3 (confirmation)
n = len(base)
raw_level_fc  = np.zeros(n, dtype=int)
raw_level_act = np.zeros(n, dtype=int)
alert_rows    = []

for i in range(n):
    row     = {
        'forecast_date': base['forecast_date'].iloc[i],
        'target_date':   base['target_date'].iloc[i],
    }
    reasons        = []
    param_level_fc = []

    for param in PARAMS:
        df_fc  = forecast_results[param]
        actual = float(df_fc['actual_1day'].iloc[i])
        fc_5d  = float(df_fc['forecast_5day'].iloc[i])

        act_st      = classify_param_status(actual, param)
        fc_thr_st   = classify_param_status(fc_5d,  param)
        cusum_alert = bool(cusum_results[param]['cusum_alert'].iloc[i])
        ewma_alert  = bool(ewma_results[param]['ewma_alert'].iloc[i])

        pi_rate = pi_results[param]['rate'][i]
        thr_p   = PI_THRESHOLDS[param]
        pi_st   = 'GREEN'
        if not np.isnan(pi_rate):
            if   pi_rate >= thr_p['red']:    pi_st = 'RED'
            elif pi_rate >= thr_p['yellow']: pi_st = 'YELLOW'

        # Sinyal level: gabungan CUSUM + EWMA + PI + forecast threshold
        # EWMA hanya berkontribusi YELLOW (tidak RED) — early warning konservatif
        # act_st sengaja TIDAK dimasukkan ke sinyal (itu ground truth, bukan prediksi)
        lv = max(
            LEVEL.get(fc_thr_st, 0),
            LEVEL.get(pi_st,     0),
            1 if cusum_alert else 0,
            1 if ewma_alert  else 0,   # FIX H3: EWMA sebagai early warning tambahan
        )
        param_level_fc.append(lv)

        parts = []
        if cusum_alert:          parts.append(f'{param}:CUSUM')
        if ewma_alert:           parts.append(f'{param}:EWMA')
        if pi_st != 'GREEN':     parts.append(f'{param}:PI={pi_rate:.2f}({pi_st})')
        if fc_thr_st != 'GREEN': parts.append(f'{param}:fc={fc_thr_st}')
        reasons.extend(parts)

        row[f'{param}_fc_status']   = REV[lv]
        row[f'{param}_act_status']  = act_st
        row[f'{param}_cusum_alert'] = cusum_alert
        row[f'{param}_ewma_alert']  = ewma_alert
        row[f'{param}_pi_rate']     = round(pi_rate, 3) if not np.isnan(pi_rate) else None
        row[f'{param}_actual']      = round(actual, 2)
        row[f'{param}_forecast_5d'] = round(fc_5d, 2)
        row[f'{param}_residual']    = round(float(cusum_results[param]['residual'].iloc[i]), 2)
        row[f'{param}_cusum_pos']   = round(float(cusum_results[param]['cusum_pos'].iloc[i]), 2)

        raw_level_act[i] = max(raw_level_act[i], LEVEL.get(act_st, 0))

    n_red    = sum(1 for lv in param_level_fc if lv >= 2)
    n_yellow = sum(1 for lv in param_level_fc if lv >= 1)

    # FIX 2: voting dilonggarkan — 1 param sudah cukup
    if   n_red    >= 2: final_lv = 2   # 2+ RED → RED
    elif n_red    == 1: final_lv = 2   # 1 RED sudah cukup → RED
    elif n_yellow >= 1: final_lv = 1   # 1 YELLOW sudah cukup → YELLOW
    else:               final_lv = 0   # semua normal → GREEN

    raw_level_fc[i] = final_lv
    row['alert_reason']   = '; '.join(reasons) if reasons else 'Normal'
    row['n_param_yellow'] = n_yellow
    row['n_param_red']    = n_red
    alert_rows.append(row)

# FIX 3: confirmation 1 hari (dari 2)
confirmed_fc = np.zeros(n, dtype=int)
for i in range(n):
    win = raw_level_fc[max(0, i - CONFIRM + 1) : i + 1]
    if len(win) >= 1 and win.max() >= 2:
        confirmed_fc[i] = 2
    elif len(win) >= 1 and win.max() >= 1:
        confirmed_fc[i] = 1
    else:
        confirmed_fc[i] = 0

for i in range(n):
    alert_rows[i]['alert_status'] = REV[confirmed_fc[i]]
    alert_rows[i]['alert_actual'] = REV[raw_level_act[i]]
    alert_rows[i]['is_critical']  = confirmed_fc[i] >= 2

df_alerts = pd.DataFrame(alert_rows)
df_alerts.to_csv(DIR_ALERTS + 'alerts_generated.csv', index=False)

print(f'\nAlerts saved: {len(df_alerts)} rows')
print('\nAlert status (forecast):')
print(df_alerts['alert_status'].value_counts().to_string())
print('\nAlert actual (ground truth):')
print(df_alerts['alert_actual'].value_counts().to_string())

with open(DIR_ALERTS + 'threshold_config.json', 'w') as fh:
    json.dump({
        'thresholds':        THRESHOLDS,
        'cusum':             {'k_factor': CUSUM_K_FACTOR, 'h_factor': CUSUM_H_FACTOR},
        'pi_thresholds':     PI_THRESHOLDS,
        'confirmation_days': CONFIRM,
        'voting_rule':       '1 param RED=RED, 1 param YELLOW=YELLOW (CUSUM conservative by design)',
        'split':             f'{int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}',
    }, fh, indent=2)

# =============================================================================
# PERFORMANCE EVALUATION
# =============================================================================

def safe_mape(actual, forecast):
    a, f = np.array(actual, dtype=float), np.array(forecast, dtype=float)
    mask = np.abs(a) > 1e-6
    if mask.sum() == 0:
        return np.nan, int((~mask).sum())
    return np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100, int((~mask).sum())


accuracy_results = {}
print('\n' + '='*70)
print(' SECTION 4.1: FORECAST ACCURACY')
print('='*70)

for param in PARAMS:
    if param not in forecast_results or forecast_results[param].empty:
        continue

    df_fc = forecast_results[param]
    label = PARAM_LABELS[param]
    unit  = PARAM_UNITS[param]

    a1, f1 = df_fc['actual_1day'].values, df_fc['forecast_1day'].values
    a5, f5 = df_fc['actual_5day'].values, df_fc['forecast_5day'].values

    mae_1d   = np.mean(np.abs(a1 - f1))
    mae_5d   = np.mean(np.abs(a5 - f5))
    rmse_1d  = np.sqrt(np.mean((a1 - f1)**2))
    rmse_5d  = np.sqrt(np.mean((a5 - f5)**2))
    mape_1d, _ = safe_mape(a1, f1)
    mape_5d, _ = safe_mape(a5, f5)

    naive_mae = naive_maes.get(param, 1.0)
    mase_1d   = mae_1d / naive_mae if naive_mae > 0 else np.nan
    mase_5d   = mae_5d / naive_mae if naive_mae > 0 else np.nan

    print(f'\n  {label} ({unit})')
    print(f'  {"─"*38}')
    print(f'  {"Metric":<8} {"1-Day":>8} {"5-Day":>8}')
    print(f'  {"MAE":<8} {mae_1d:>8.3f} {mae_5d:>8.3f}  {unit}')
    print(f'  {"RMSE":<8} {rmse_1d:>8.3f} {rmse_5d:>8.3f}  {unit}')
    print(f'  {"MAPE":<8} {mape_1d:>7.2f}% {mape_5d:>7.2f}%')
    print(f'  {"MASE":<8} {mase_1d:>8.3f} {mase_5d:>8.3f}  '
          f'({"< 1.0 beats naive" if mase_1d < 1 else "> 1.0 below naive"})')

    accuracy_results[param] = {
        'MAE_1day':    round(mae_1d,  4), 'MAE_5day':    round(mae_5d,  4),
        'RMSE_1day':   round(rmse_1d, 4), 'RMSE_5day':   round(rmse_5d, 4),
        'MAPE_1day':   round(mape_1d, 2), 'MAPE_5day':   round(mape_5d, 2),
        'MASE_1day':   round(mase_1d, 3), 'MASE_5day':   round(mase_5d, 3),
        'naive_mae':   round(naive_mae, 4),
        'n_forecasts': len(df_fc),
    }


ALERT_CLASSES = ['OFF', 'GREEN', 'YELLOW', 'RED']
y_pred  = df_alerts['alert_status'].values
y_true  = df_alerts['alert_actual'].values
n_total = len(df_alerts)

print('\n' + '='*70)
print(' SECTION 4.2: ALERT CLASSIFICATION PERFORMANCE')
print('='*70)

confusion = {
    fc: {ac: int(((y_pred == fc) & (y_true == ac)).sum()) for ac in ALERT_CLASSES}
    for fc in ALERT_CLASSES
}

print(f'\n  Total test points: {n_total}')
print()
print(classification_report(y_true, y_pred, labels=['GREEN', 'YELLOW', 'RED'], zero_division=0))

# Confusion matrix text
print('  CONFUSION MATRIX (rows=forecast, cols=actual):')
header = f"  {'':>10}" + "".join(f"{'Act:'+c:>10}" for c in ALERT_CLASSES)
print(header)
print('  ' + '─' * 50)
for fc in ALERT_CLASSES:
    row_str = f"  {'Fc:'+fc:>10}" + "".join(f"{confusion[fc][ac]:>10}" for ac in ALERT_CLASSES)
    print(row_str)
print()

class_metrics = {}
for cls in ALERT_CLASSES:
    tp = confusion[cls].get(cls, 0)
    fp = sum(confusion[cls].get(o, 0) for o in ALERT_CLASSES if o != cls)
    fn = sum(confusion[fc].get(cls, 0) for fc in ALERT_CLASSES if fc != cls)
    support  = tp + fn
    prec     = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    rec      = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1       = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    class_metrics[cls] = {
        'precision': round(prec, 1), 'recall': round(rec, 1), 'f1': round(f1, 1),
        'tp': tp, 'fp': fp, 'fn': fn, 'support': support,
    }

correct  = sum(confusion[c][c] for c in ALERT_CLASSES)
accuracy = correct / n_total * 100
classification_results = {
    'confusion_matrix': confusion,
    'class_metrics':    class_metrics,
    'accuracy':         round(accuracy, 1),
    'n_total':          n_total,
}


# Lead time
print('='*70)
print(' SECTION 4.2b: LEAD TIME ANALYSIS')
print('='*70)

def compute_lead_time(df_a, target_class='RED'):
    df_a = df_a.copy()
    df_a['target_date'] = pd.to_datetime(df_a['target_date'])
    df_a['is_target']   = df_a['alert_actual'] == target_class

    events = []
    curr   = None
    for i, row in df_a.iterrows():
        if row['is_target']:
            if curr is None:
                curr = {'start_idx': i, 'start_date': row['target_date'], 'days': 1}
            else:
                curr['days'] += 1
        else:
            if curr:
                events.append(curr)
                curr = None
    if curr:
        events.append(curr)

    print(f'\n  {target_class} event blocks: {len(events)}')
    if not events:
        return None

    lead_times = []
    for event in events:
        si    = event['start_idx']
        found = False; lt = 0
        for j in range(si - 1, max(-1, si - 15), -1):
            st = df_a.loc[j, 'alert_status'] if j in df_a.index else 'GREEN'
            if st in ('YELLOW', 'RED'):
                lt = si - j; found = True
            else:
                break
        if not found:
            at_event = df_a.loc[si, 'alert_status'] if si in df_a.index else 'GREEN'
            if at_event in ('YELLOW', 'RED'):
                lt, found = 0, True

        icon = '✓' if found else '✗'
        lt_s = f'{lt}d' if found else 'MISSED'
        print(f'    {icon} {event["start_date"].date()} ({event["days"]}d {target_class}) → Lead time: {lt_s}')
        lead_times.append({'detected': found, 'lead_time_days': lt if found else None})

    n_det  = sum(1 for lt in lead_times if lt['detected'])
    lt_vals = [lt['lead_time_days'] for lt in lead_times if lt['detected']]

    print(f'\n  Detected: {n_det}/{len(lead_times)}')
    if lt_vals:
        mean_lt = np.mean(lt_vals)
        print(f'  Mean lead time: {mean_lt:.1f} days  (target >= {TARGETS["Lead_Time_Avg"]} days)  '
              f'{"MET" if mean_lt >= TARGETS["Lead_Time_Avg"] else "NOT MET"}')
        print(f'  Min / Max     : {min(lt_vals)} / {max(lt_vals)} days')

    result = {'target_class': target_class, 'n_events': len(lead_times),
              'n_detected': n_det, 'n_missed': len(lead_times) - n_det, 'events': lead_times}
    if lt_vals:
        result.update({
            'mean_lead_time':   round(np.mean(lt_vals), 1),
            'median_lead_time': round(np.median(lt_vals), 1),
            'min_lead_time':    int(min(lt_vals)),
            'max_lead_time':    int(max(lt_vals)),
        })
    return result


lead_time_results = compute_lead_time(df_alerts, 'RED')
if not lead_time_results or lead_time_results.get('n_events', 0) == 0:
    print('\n  No RED events — trying YELLOW...')
    lead_time_results = compute_lead_time(df_alerts, 'YELLOW')

# =============================================================================
# PLOTS
# =============================================================================

# Alert timeline
alert_colors = {'GREEN': '#22C55E', 'YELLOW': '#F59E0B', 'RED': '#EF4444', 'OFF': '#9CA3AF'}
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True,
                          gridspec_kw={'height_ratios': [2, 2, 2, 1]})

for ax, param in zip(axes[:3], PARAMS):
    df_fc = forecast_results[param]
    dates = df_fc['target_date_5day'] if 'target_date_5day' in df_fc else df_fc['target_date']

    ax.plot(dates, df_fc['actual_5day'],   color='#1E40AF', linewidth=1, label='Actual at 5-day target', zorder=3)
    ax.plot(dates, df_fc['forecast_5day'], color='#93C5FD', linewidth=1, linestyle='--',
            label='SARIMA 5-day forecast', zorder=2)
    ax.fill_between(dates, df_fc['pi_lower_5day'], df_fc['pi_upper_5day'],
                    alpha=0.15, color='#3B82F6', label='95% PI', zorder=1)

    thr = THRESHOLDS.get(param, {})
    if param == 'discharge_temp':
        ax.axhline(thr.get('YELLOW', 150), color='orange', linestyle=':', linewidth=1,
                   label=f'YELLOW limit ({thr.get("YELLOW")}°F)')
        ax.axhline(thr.get('RED', 160), color='red', linestyle=':', linewidth=1,
                   label=f'RED limit ({thr.get("RED")}°F)')

    ax.set_ylabel(f"{PARAM_LABELS[param]}\n({PARAM_UNITS[param]})", fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

ax = axes[3]
for i, row in df_alerts.iterrows():
    ax.barh(1, 1, left=i, color=alert_colors.get(row['alert_status'], 'gray'), height=0.4, align='center')
    ax.barh(0, 1, left=i, color=alert_colors.get(row['alert_actual'], 'gray'),  height=0.4, align='center')
ax.set_yticks([0, 1])
ax.set_yticklabels(['Ground Truth', 'System Alert'], fontsize=8)
ax.set_xlabel('Test Period (days since start)')
patches = [mpatches.Patch(color=c, label=l) for l, c in alert_colors.items()]
ax.legend(handles=patches, loc='upper left', fontsize=7, ncol=4)

fig.suptitle('VRU Alert Timeline — Improved System (SARIMA + CUSUM + PI)', fontsize=12)
plt.tight_layout()
plt.savefig(DIR_ALERTS + 'alert_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\nAlert timeline saved')

# CUSUM plot
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
for ax, param in zip(axes, PARAMS):
    cs    = cusum_results[param]
    df_fc = forecast_results[param]
    dates = df_fc['target_date']
    h_val = float(cs['cusum_h'].iloc[0])

    ax.plot(dates, cs['cusum_pos'], color='#EF4444', linewidth=1, label='C+ (upward drift)')
    ax.plot(dates, cs['cusum_neg'], color='#3B82F6', linewidth=1, label='C- (downward drift)')
    ax.axhline(h_val, color='black', linestyle='--', linewidth=1.2,
               label=f'Decision threshold h={h_val:.1f}')
    for _, row in df_alerts.iterrows():
        if row['alert_actual'] == 'RED':    ax.axvline(row['target_date'], color='red',    alpha=0.3, linewidth=1.5)
        elif row['alert_actual'] == 'YELLOW': ax.axvline(row['target_date'], color='orange', alpha=0.2, linewidth=1)

    ax.set_ylabel(f'CUSUM\n{PARAM_LABELS[param][:12]}', fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

axes[0].set_title('CUSUM Control Chart on ARIMA Residuals\n'
                  '(Page 1954 | Venkatasubramanian et al. 2003)\n'
                  'Red vertical = actual RED, Orange = actual YELLOW', fontsize=11)
axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig(DIR_PERF + 'cusum_monitoring.png', dpi=150, bbox_inches='tight')
plt.close()

# PI violation rate plot
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
for ax, param in zip(axes, PARAMS):
    df_fc  = forecast_results[param]
    dates  = df_fc['target_date']
    rate   = pi_results[param]['rate']

    ax.plot(dates, rate, color='#6366F1', linewidth=1.2, label='PI violation rate (7-day)')
    ax.axhline(PI_YELLOW_RATE, color='orange', linestyle='--', linewidth=1,
               label=f'YELLOW threshold ({PI_YELLOW_RATE:.2f})')
    ax.axhline(PI_RED_RATE, color='red', linestyle='--', linewidth=1,
               label=f'RED threshold ({PI_RED_RATE:.2f})')
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel(f'Violation Rate\n{PARAM_LABELS[param][:12]}', fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    for _, row in df_alerts.iterrows():
        if row['alert_actual'] == 'RED':
            ax.axvline(row['target_date'], color='red', alpha=0.3, linewidth=1.5)

axes[0].set_title('Prediction Interval Violation Rate (Rolling 7-day)\n'
                  '(Chatfield 1993 | Sikorska et al. 2011)', fontsize=11)
axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig(DIR_PERF + 'pi_violation_rate.png', dpi=150, bbox_inches='tight')
plt.close()

# Forecast accuracy bar chart
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
params = list(accuracy_results.keys())
labels = [p.replace('_', '\n').title() for p in params]
x = np.arange(len(params)); w = 0.35

for ax, metric, title in zip(axes, ['MAE', 'RMSE', 'MAPE', 'MASE'], ['MAE', 'RMSE', 'MAPE (%)', 'MASE']):
    v1 = [accuracy_results[p][f'{metric}_1day'] for p in params]
    v5 = [accuracy_results[p][f'{metric}_5day'] for p in params]
    ax.bar(x - w/2, v1, w, label='1-Day', color='#3B82F6', alpha=0.85)
    ax.bar(x + w/2, v5, w, label='5-Day', color='#EF4444', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    if metric == 'MASE':
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1.2, label='Naive baseline')
        ax.legend(fontsize=7)

fig.suptitle('Section 4.1: SARIMA Forecast Accuracy (with Box-Cox)', fontsize=12)
plt.tight_layout()
plt.savefig(DIR_PERF + 'forecast_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()

# Confusion matrix heatmap
matrix = np.array([[confusion[fc].get(ac, 0) for ac in ALERT_CLASSES] for fc in ALERT_CLASSES])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(matrix, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(ALERT_CLASSES))); ax.set_yticks(range(len(ALERT_CLASSES)))
ax.set_xticklabels(ALERT_CLASSES); ax.set_yticklabels(ALERT_CLASSES)
ax.set_xlabel('Actual', fontsize=11); ax.set_ylabel('Forecast', fontsize=11)
ax.set_title('Alert Classification Confusion Matrix\n(Improved: CUSUM + PI + Threshold)', fontsize=11)
for i in range(len(ALERT_CLASSES)):
    for j in range(len(ALERT_CLASSES)):
        val   = matrix[i, j]
        color = 'white' if val > matrix.max() * 0.6 else 'black'
        ax.text(j, i, str(val), ha='center', va='center', color=color, fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(DIR_PERF + 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print('All plots saved')
from vru_evaluation import run_full_evaluation
run_full_evaluation(forecast_results, df_alerts, naive_maes, DIR_PERF, TARGETS)

# =============================================================================
# HYPOTHESIS VALIDATION & SAVE JSON
# =============================================================================

print('\n' + '='*70)
print(' HYPOTHESIS VALIDATION')
print('='*70)

h_count, h_pass = 0, 0

if accuracy_results:
    avg_mape_1d = np.mean([accuracy_results[p]['MAPE_1day'] for p in accuracy_results])
    avg_mape_5d = np.mean([accuracy_results[p]['MAPE_5day'] for p in accuracy_results])
    avg_mase    = np.mean([accuracy_results[p]['MASE_1day'] for p in accuracy_results])
    h1 = avg_mape_1d < TARGETS['MAPE_1day'] and avg_mape_5d < TARGETS['MAPE_5day']
    print(f'  H1 Forecast Accuracy  : {"MET" if h1 else "NOT MET":>10}  '
          f'(MAPE_1d={avg_mape_1d:.1f}% MAPE_5d={avg_mape_5d:.1f}% MASE={avg_mase:.3f})')
    h_count += 1; h_pass += int(h1)

if classification_results:
    acc    = classification_results['accuracy']
    h2     = acc > 75
    red_r  = class_metrics.get('RED',    {}).get('recall', 0)
    yel_r  = class_metrics.get('YELLOW', {}).get('recall', 0)
    print(f'  H2 Alert Classification: {"MET" if h2 else "NOT MET":>10}  '
          f'(accuracy={acc:.1f}% RED_recall={red_r:.1f}% YELLOW_recall={yel_r:.1f}%)')
    h_count += 1; h_pass += int(h2)

if lead_time_results and lead_time_results.get('n_detected', 0) > 0:
    mean_lt = lead_time_results['mean_lead_time']
    h3 = mean_lt >= TARGETS['Lead_Time_Avg']
    print(f'  H3 Lead Time          : {"MET" if h3 else "NOT MET":>10}  '
          f'(mean={mean_lt:.1f} days, target>={TARGETS["Lead_Time_Avg"]})')
    h_count += 1; h_pass += int(h3)
else:
    print('  H3 Lead Time          :  INSUFFICIENT DATA')

print(f'\n  OVERALL: {h_pass}/{h_count} hypotheses validated')
print('='*70)

def json_convert(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, pd.Timestamp):   return str(obj)
    return str(obj)

all_results = {
    'forecast_accuracy': accuracy_results,
    'classification': {
        'confusion_matrix': classification_results.get('confusion_matrix'),
        'class_metrics':    classification_results.get('class_metrics'),
        'accuracy':         classification_results.get('accuracy'),
        'n_total':          classification_results.get('n_total'),
    },
    'lead_time': {k: v for k, v in (lead_time_results or {}).items() if k != 'events'},
    'targets': TARGETS,
    'method': 'SARIMA + Box-Cox + CUSUM + PI_violation_rate',
    'split': f'{int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}',
    'references': [
        'Page (1954) Biometrika 41 — CUSUM',
        'Lucas & Saccucci (1990) Technometrics 32 — EWMA',
        'Box & Cox (1964) JRSS-B 26 — Box-Cox transformation',
        'Venkatasubramanian et al. (2003) Computers & ChemEng 27 — residual fault detection',
        'Chatfield (1993) JBES 11 — prediction intervals',
        'Sikorska et al. (2011) MSSP 25 — industrial prognostics',
    ],
}

with open(DIR_PERF + 'performance_metrics.json', 'w') as fh:
    json.dump(all_results, fh, indent=2, default=json_convert)
print(f'\nResults saved: {DIR_PERF}performance_metrics.json')

# =============================================================================
# ZIP OUTPUT
# =============================================================================

output_zip = OUT_ROOT + 'output.zip'
with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(OUT_ROOT):
        for file in files:
            if not file.endswith('.zip'):
                full_path = os.path.join(root, file)
                arcname   = os.path.relpath(full_path, OUT_ROOT)
                zf.write(full_path, arcname)

size_kb = os.path.getsize(output_zip) / 1024
print(f'Zipped: {output_zip} ({size_kb:.1f} KB)')
print('\nDONE')
