import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, json, os, zipfile

warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.metrics import classification_report
from pmdarima import auto_arima

# =============================================================================
# PATHS
# =============================================================================
DATA_FILE  = 'vru_data_full_4years.csv'
FALLBACK_DATA_FILE = 'vru_preprocessed.csv'
OUT_ROOT   = 'output_baseline_sarima/'
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

TRAIN_RATIO    = 0.80
HORIZON        = 5
REFIT_INTERVAL = 20

THRESHOLDS = {
    'discharge_temp':     {'OFF': 90, 'YELLOW': 150, 'RED': 160},
    'discharge_pressure': {'LOW': 10, 'HIGH': 30},
    'jacket_water':       {'LOW': 12, 'HIGH': 20},
}

ALERT_RULES = {
    'discharge_temp': {
        'OFF': 90,
        'NORMAL': (110, 150),
        'WARNING': 150,
        'CRITICAL': 300,
    },
    'discharge_pressure': {
        'NORMAL': (10, 30),
        'LOW': 10,
        'HIGH': 30,
    },
    'jacket_water': {
        'NORMAL': (12, 20),
        'LOW': 12,
        'HIGH': 20,
    },
}

CONFIRM = 1

TARGETS = {
    'MAPE_1day':    10.0,
    'MAPE_5day':    15.0,
    'MASE':          1.0,
    'Lead_Time_Avg': 3.0,
}

LEVEL = {'GREEN': 0, 'YELLOW': 1, 'RED': 2, 'OFF': -1}
REV   = {0: 'GREEN', 1: 'YELLOW', 2: 'RED', -1: 'OFF'}

print('=== BASELINE SARIMA SYSTEM (Simple Threshold, NO CUSUM/EWMA) ===')
print(f'  Forecast : SARIMA + Box-Cox')
print(f'  Alert    : Simple threshold ONLY (no CUSUM/EWMA/PI)')
print(f'  Output   : {OUT_ROOT}')

# =============================================================================
# LOAD & PREPROCESS
# =============================================================================
if not os.path.exists(DATA_FILE) and os.path.exists(FALLBACK_DATA_FILE):
    print(f'\nData file not found: {DATA_FILE} - using {FALLBACK_DATA_FILE}')
    DATA_FILE = FALLBACK_DATA_FILE

df_full = pd.read_csv(DATA_FILE, parse_dates=['date'])
df_full = df_full.sort_values('date').reset_index(drop=True)

df = df_full.copy().reset_index(drop=True)  # down-days already removed in preprocessing
df = df.set_index('date')
full_idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
df = df.reindex(full_idx)
for col in PARAMS:
    df[col] = df[col].ffill(limit=3)
df = df.reset_index().rename(columns={'index': 'date'})

df_valid = df.dropna(subset=PARAMS).reset_index(drop=True)
split_idx = int(len(df_valid) * TRAIN_RATIO)
train = df_valid.iloc[:split_idx].copy()
test  = df_valid.iloc[split_idx:].copy().reset_index(drop=True)

print(f'\n  Full dataset : {len(df_full)} rows')
print(f'  Train        : {len(train)} rows  ({train.date.min().date()} to {train.date.max().date()})')
print(f'  Test         : {len(test)} rows   ({test.date.min().date()} to {test.date.max().date()})')

# =============================================================================
# SARIMA ORDER DETECTION
# =============================================================================
def detect_sarima_order(series, param_name):
    print(f'  Running auto_arima for {param_name}...')
    model = auto_arima(
        series.dropna(),
        seasonal=True, m=12,
        d=None, D=None,
        max_p=3, max_q=3,
        max_P=2, max_Q=2,
        start_p=0, start_q=0,
        start_P=0, start_Q=0,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
    )
    order = model.order
    seasonal_order = model.seasonal_order
    print(f'  → SARIMA{order}x{seasonal_order}')
    return order, seasonal_order


def fit_sarima_boxcox(series, order, seasonal_order):
    shift = 0
    if series.min() <= 0:
        shift = abs(series.min()) + 1.0
    s_shifted = series + shift
    transformed, lam = boxcox(s_shifted.dropna())
    ts_transformed = pd.Series(transformed, index=series.dropna().index)
    model = SARIMAX(
        ts_transformed, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=200)
    return fit, lam, shift


def fit_sarima_raw(series, order, seasonal_order):
    model = SARIMAX(
        series.dropna(),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=200)


def forecast_from_raw_fit(raw_fit, horizon):
    fc_obj = raw_fit.get_forecast(steps=horizon)
    fc_mean = fc_obj.predicted_mean.values
    fc_ci = fc_obj.conf_int(alpha=0.05)
    return fc_mean, fc_ci.iloc[:, 0].values, fc_ci.iloc[:, 1].values


def forecast_is_plausible(fc_mean, history_vals, param):
    if not np.all(np.isfinite(fc_mean)):
        return False
    hist = np.asarray(history_vals, dtype=float)
    hist = hist[np.isfinite(hist)]
    if len(hist) == 0:
        return True
    hist_median = float(np.median(hist[-min(len(hist), 120):]))
    if param == 'discharge_temp' and hist_median >= THRESHOLDS['discharge_temp']['OFF']:
        return np.nanmedian(fc_mean) >= THRESHOLDS['discharge_temp']['OFF']
    return True


print('\nSARIMA ORDER DETECTION')
print('─' * 60)
orders_all = {}
for param in PARAMS:
    print(f'\n  {PARAM_LABELS[param]}:')
    o, so = detect_sarima_order(train[param], param)
    orders_all[param] = {'order': o, 'seasonal_order': so}

# Naive MAE
naive_maes = {}
for param in PARAMS:
    series = df_valid[param].values
    naive_maes[param] = float(np.mean(np.abs(np.diff(series))))
    print(f'  Naive MAE [{param}]: {naive_maes[param]:.4f}')

# =============================================================================
# ROLLING WALK-FORWARD FORECAST
# =============================================================================
print('\nStarting rolling walk-forward forecast...')
print(f'  Test points  : {len(test)}')

forecast_results = {}
fallback_counts = {}
n_test = len(test)

for param in PARAMS:
    print(f'\n--- {PARAM_LABELS[param]} ---')
    o  = orders_all[param]['order']
    so = orders_all[param]['seasonal_order']

    records      = []
    fit          = None
    raw_fit      = None
    lam_bc       = None
    shift_bc     = None
    history_vals = list(train[param].values)
    fallback_counts[param] = 0

    for i in range(n_test - HORIZON):
        fc_date = test.iloc[i]['date']

        if fit is None or i % REFIT_INTERVAL == 0:
            try:
                hist_s = pd.Series(history_vals, dtype=float)
                fit, lam_bc, shift_bc = fit_sarima_boxcox(hist_s, o, so)
                raw_fit = fit_sarima_raw(hist_s, o, so)
                print(f'  [refit] step {i:3d}/{n_test - HORIZON}  ({fc_date.date()})  AIC={fit.aic:.1f}')
            except Exception as exc:
                print(f'  [refit FAILED step {i}]: {exc}')
                fit = None
                raw_fit = None
                history_vals.append(float(test.iloc[i][param]))
                continue

        try:
            fc_obj     = fit.get_forecast(steps=HORIZON)
            fc_mean_bc = fc_obj.predicted_mean.values
            fc_ci_bc   = fc_obj.conf_int(alpha=0.05)

            fc_mean  = inv_boxcox(fc_mean_bc,                 lam_bc) - shift_bc
            ci_lower = inv_boxcox(fc_ci_bc.iloc[:, 0].values, lam_bc) - shift_bc
            ci_upper = inv_boxcox(fc_ci_bc.iloc[:, 1].values, lam_bc) - shift_bc

            if not forecast_is_plausible(fc_mean, history_vals, param):
                if raw_fit is None:
                    raw_fit = fit_sarima_raw(pd.Series(history_vals, dtype=float), o, so)
                fc_mean, ci_lower, ci_upper = forecast_from_raw_fit(raw_fit, HORIZON)
                fallback_counts[param] += 1

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

    df_fc = pd.DataFrame(records)
    if not df_fc.empty:
        df_fc['forecast_date'] = pd.to_datetime(df_fc['forecast_date'])
        df_fc['target_date']   = pd.to_datetime(df_fc['target_date'])
        df_fc['target_date_1day'] = pd.to_datetime(df_fc['target_date_1day'])
        df_fc['target_date_5day'] = pd.to_datetime(df_fc['target_date_5day'])
        df_fc.to_csv(DIR_MODELS + f'forecast_{param}.csv', index=False)
        df_fc.to_csv(DIR_MODELS + f'forecasts_{param}.csv', index=False)
        print(f'  {len(df_fc)} records saved')
    forecast_results[param] = df_fc
    if fallback_counts[param]:
        print(f'  Raw-scale SARIMA fallback used for {fallback_counts[param]} forecast windows')

print('\nForecast done.')

with open(DIR_MODELS + 'model_metadata.json', 'w') as fh:
    json.dump({
        'model': 'SARIMA + Box-Cox (baseline)',
        'orders': {
            param: {
                'order': list(order_data['order']),
                'seasonal_order': list(order_data['seasonal_order']),
            }
            for param, order_data in orders_all.items()
        },
        'naive_mae': naive_maes,
        'raw_fallback_counts': fallback_counts,
        'split': f'{int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}',
        'horizon': HORIZON,
        'refit_interval': REFIT_INTERVAL,
    }, fh, indent=2)

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
fig.suptitle('S2 Baseline SARIMA - 5-Day Forecast vs Actual', fontsize=12)
plt.tight_layout()
plt.savefig(DIR_PLOTS + 'forecast_vs_actual_5day.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"5-day forecast plot saved: {DIR_PLOTS}forecast_vs_actual_5day.png")

# =============================================================================
# SIMPLE THRESHOLD ALERT (no CUSUM/EWMA/PI)
# =============================================================================
def check_discharge_temp(value):
    rules = ALERT_RULES['discharge_temp']
    if value < rules['OFF']:
        return False, False, 'OFF'
    if value > rules['CRITICAL']:
        return True, True, f'CRITICAL: Temp {value:.1f} {PARAM_UNITS["discharge_temp"]} (>{rules["CRITICAL"]})'
    if value > rules['WARNING']:
        return True, False, f'Temp HIGH {value:.1f} {PARAM_UNITS["discharge_temp"]} (>{rules["WARNING"]})'
    lo, hi = rules['NORMAL']
    if lo <= value <= hi:
        return False, False, f'Temp OK {value:.1f} {PARAM_UNITS["discharge_temp"]} ({lo}-{hi})'
    return False, False, f'Temp {value:.1f} {PARAM_UNITS["discharge_temp"]} (startup/warmup zone)'


def check_discharge_pressure(value):
    rules = ALERT_RULES['discharge_pressure']
    lo, hi = rules['NORMAL']
    if value < rules['LOW']:
        return True, False, f'Press LOW {value:.1f} psi (<{rules["LOW"]})'
    if value > rules['HIGH']:
        return True, False, f'Press HIGH {value:.1f} psi (>{rules["HIGH"]})'
    return False, False, f'Press OK {value:.1f} psi ({lo}-{hi})'


def check_jacket_water(value):
    rules = ALERT_RULES['jacket_water']
    lo, hi = rules['NORMAL']
    if value < rules['LOW']:
        return True, False, f'Jacket LOW {value:.1f} psi (<{rules["LOW"]}) - leading indicator'
    if value > rules['HIGH']:
        return True, False, f'Jacket HIGH {value:.1f} psi (>{rules["HIGH"]})'
    return False, False, f'Jacket OK {value:.1f} psi ({lo}-{hi})'


def classify_alert(temp_value, pressure_value, jacket_value):
    if temp_value < ALERT_RULES['discharge_temp']['OFF']:
        return {
            'status': 'OFF',
            'violations': [],
            'n_violations': 0,
            'is_critical': False,
            'details': {
                'discharge_temp': 'OFF',
                'discharge_pressure': 'N/A',
                'jacket_water': 'N/A',
            },
            'reason': f'Compressor not running (temp {temp_value:.1f} < {ALERT_RULES["discharge_temp"]["OFF"]})',
        }

    temp_violation, temp_critical, temp_desc = check_discharge_temp(temp_value)
    press_violation, press_critical, press_desc = check_discharge_pressure(pressure_value)
    jacket_violation, jacket_critical, jacket_desc = check_jacket_water(jacket_value)

    violations = []
    is_critical = False
    for is_violation, is_crit, desc in [
        (temp_violation, temp_critical, temp_desc),
        (press_violation, press_critical, press_desc),
        (jacket_violation, jacket_critical, jacket_desc),
    ]:
        if is_violation:
            violations.append(desc)
            is_critical = is_critical or is_crit

    n_violations = len(violations)
    if is_critical:
        status = 'RED'
        reason = f'CRITICAL violation: {violations[0]}'
    elif n_violations >= 2:
        status = 'RED'
        reason = f'{n_violations} parameters out of range'
    elif n_violations == 1:
        status = 'YELLOW'
        reason = violations[0]
    else:
        status = 'GREEN'
        reason = 'All parameters within normal operating ranges'

    return {
        'status': status,
        'violations': violations,
        'n_violations': n_violations,
        'is_critical': is_critical,
        'details': {
            'discharge_temp': 'CRITICAL' if temp_critical else ('VIOLATION' if temp_violation else 'OK'),
            'discharge_pressure': 'VIOLATION' if press_violation else 'OK',
            'jacket_water': 'VIOLATION' if jacket_violation else 'OK',
        },
        'reason': reason,
    }


base = forecast_results[PARAMS[0]][['forecast_date', 'target_date']].copy()
n    = len(base)
raw_level_fc = np.zeros(n, dtype=int)
alert_rows   = []

for i in range(n):
    temp_fc_5d   = float(forecast_results['discharge_temp']['forecast_5day'].iloc[i])
    press_fc_5d  = float(forecast_results['discharge_pressure']['forecast_5day'].iloc[i])
    jacket_fc_5d = float(forecast_results['jacket_water']['forecast_5day'].iloc[i])

    temp_fc_1d   = float(forecast_results['discharge_temp']['forecast_1day'].iloc[i])
    press_fc_1d  = float(forecast_results['discharge_pressure']['forecast_1day'].iloc[i])
    jacket_fc_1d = float(forecast_results['jacket_water']['forecast_1day'].iloc[i])

    temp_actual   = float(forecast_results['discharge_temp']['actual_5day'].iloc[i])
    press_actual  = float(forecast_results['discharge_pressure']['actual_5day'].iloc[i])
    jacket_actual = float(forecast_results['jacket_water']['actual_5day'].iloc[i])

    result_5d     = classify_alert(temp_fc_5d, press_fc_5d, jacket_fc_5d)
    result_1d     = classify_alert(temp_fc_1d, press_fc_1d, jacket_fc_1d)
    result_actual = classify_alert(temp_actual, press_actual, jacket_actual)

    raw_level_fc[i] = LEVEL.get(result_5d['status'], 0)

    alert_rows.append({
        'forecast_date': base['forecast_date'].iloc[i],
        'target_date': base['target_date'].iloc[i],
        'alert_status_raw': result_5d['status'],
        'alert_status': result_5d['status'],
        'alert_reason': result_5d['reason'],
        'n_violations': result_5d['n_violations'],
        'is_critical': result_5d['is_critical'],
        'temp_status': result_5d['details']['discharge_temp'],
        'pressure_status': result_5d['details']['discharge_pressure'],
        'jacket_status': result_5d['details']['jacket_water'],
        'alert_1day': result_1d['status'],
        'alert_actual': result_actual['status'],
        'actual_reason': result_actual['reason'],
        'temp_forecast_5d': round(temp_fc_5d, 2),
        'press_forecast_5d': round(press_fc_5d, 2),
        'jacket_forecast_5d': round(jacket_fc_5d, 2),
        'temp_forecast_1d': round(temp_fc_1d, 2),
        'press_forecast_1d': round(press_fc_1d, 2),
        'jacket_forecast_1d': round(jacket_fc_1d, 2),
        'temp_actual': round(temp_actual, 2),
        'press_actual': round(press_actual, 2),
        'jacket_actual': round(jacket_actual, 2),
        'violations': '; '.join(result_5d['violations']) if result_5d['violations'] else '',
    })

# Confirmation window
confirmed_fc = np.zeros(n, dtype=int)
for i in range(n):
    win = raw_level_fc[max(0, i - CONFIRM + 1) : i + 1]
    if win.max() >= 2:
        confirmed_fc[i] = 2
    elif win.max() >= 1:
        confirmed_fc[i] = 1
    elif win.min() < 0:
        confirmed_fc[i] = -1
    else:
        confirmed_fc[i] = 0

for i in range(n):
    alert_rows[i]['alert_status'] = REV[confirmed_fc[i]]
    alert_rows[i]['is_critical']  = confirmed_fc[i] >= 2

df_alerts = pd.DataFrame(alert_rows)
df_alerts.to_csv(DIR_ALERTS + 'alerts_generated.csv', index=False)

threshold_config = {}
for param, thresholds in ALERT_RULES.items():
    threshold_config[param] = {}
    for key, val in thresholds.items():
        threshold_config[param][key] = list(val) if isinstance(val, tuple) else val
with open(DIR_ALERTS + 'threshold_config.json', 'w') as fh:
    json.dump({
        'thresholds': threshold_config,
        'confirmation_days': CONFIRM,
        'primary_alert_horizon': '5-day forecast',
        'classification_rule': 'OFF if temp < 90; RED if critical temp > 300 or 2+ violations; YELLOW if 1 violation; GREEN otherwise',
        'baseline_note': 'S2 uses SARIMA point forecast thresholds only; no CUSUM, EWMA, or PI violation layer.',
    }, fh, indent=2)

print(f'\nAlerts saved: {len(df_alerts)} rows')
print('\nAlert status (forecast):')
print(df_alerts['alert_status'].value_counts().to_string())
print('\nAlert actual (ground truth):')
print(df_alerts['alert_actual'].value_counts().to_string())

# =============================================================================
# PERFORMANCE EVALUATION
# =============================================================================
def safe_mape(actual, forecast):
    a, f = np.array(actual, dtype=float), np.array(forecast, dtype=float)
    mask = np.abs(a) > 1e-6
    if mask.sum() == 0:
        return np.nan, 0
    return np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100, int((~mask).sum())


accuracy_results = {}
mape_all = []

print('\n' + '='*70)
print(' FORECAST ACCURACY (BASELINE SARIMA)')
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

    mape_all.append(mape_1d)

    print(f'\n  {label} ({unit})')
    print(f'  {"─"*38}')
    print(f'  {"Metric":<8} {"1-Day":>8} {"5-Day":>8}')
    print(f'  {"MAE":<8} {mae_1d:>8.3f} {mae_5d:>8.3f}  {unit}')
    print(f'  {"RMSE":<8} {rmse_1d:>8.3f} {rmse_5d:>8.3f}  {unit}')
    print(f'  {"MAPE":<8} {mape_1d:>7.2f}% {mape_5d:>7.2f}%')
    print(f'  {"MASE":<8} {mase_1d:>8.3f} {mase_5d:>8.3f}  ({"< 1.0 beats naive" if mase_1d < 1 else "> 1.0 below naive"})')

    accuracy_results[param] = {
        'MAE_1day':   round(mae_1d, 4),  'MAE_5day':   round(mae_5d, 4),
        'RMSE_1day':  round(rmse_1d, 4), 'RMSE_5day':  round(rmse_5d, 4),
        'MAPE_1day':  round(mape_1d, 2), 'MAPE_5day':  round(mape_5d, 2),
        'MASE_1day':  round(mase_1d, 3), 'MASE_5day': round(mase_5d, 3),
        'naive_mae':  round(naive_mae, 4),
        'n_forecasts': len(df_fc),
    }

avg_mape = float(np.mean(mape_all))

# Alert classification
y_pred  = df_alerts['alert_status'].values
y_true  = df_alerts['alert_actual'].values
n_total = len(df_alerts)

print('\n' + '='*70)
print(' ALERT CLASSIFICATION PERFORMANCE (BASELINE SARIMA)')
print('='*70)
print(f'\n  Total test points: {n_total}\n')
print(classification_report(y_true, y_pred, labels=['GREEN', 'YELLOW', 'RED'], zero_division=0))

ALERT_CLASSES = ['OFF', 'GREEN', 'YELLOW', 'RED']
confusion = {
    fc: {ac: int(((y_pred == fc) & (y_true == ac)).sum()) for ac in ALERT_CLASSES}
    for fc in ALERT_CLASSES
}

print('  CONFUSION MATRIX (rows=forecast, cols=actual):')
header = f"  {'Forecast / Actual':>22}" + ''.join(f"{'Act:'+c:>10}" for c in ALERT_CLASSES) + f"{'Total':>8}"
print(header)
print('  ' + '-' * 62)
for fc in ALERT_CLASSES:
    row_total = sum(confusion[fc][ac] for ac in ALERT_CLASSES)
    row_str = f"  {'Fc:'+fc:>22}" + ''.join(f"{confusion[fc][ac]:>10}" for ac in ALERT_CLASSES) + f"{row_total:>8}"
    print(row_str)
col_total = f"  {'Total':>22}" + ''.join(f"{sum(confusion[fc][ac] for fc in ALERT_CLASSES):>10}" for ac in ALERT_CLASSES) + f"{n_total:>8}"
print('  ' + '-' * 62)
print(col_total)
print()

class_metrics = {}
for cls in ALERT_CLASSES:
    tp = confusion[cls].get(cls, 0)
    fp = sum(confusion[cls].get(other, 0) for other in ALERT_CLASSES if other != cls)
    fn = sum(confusion[fc].get(cls, 0) for fc in ALERT_CLASSES if fc != cls)
    support = tp + fn
    predicted = tp + fp
    precision = tp / predicted * 100 if predicted > 0 else 0.0
    recall = tp / support * 100 if support > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    class_metrics[cls] = {
        'precision': round(precision, 1),
        'recall': round(recall, 1),
        'f1': round(f1, 1),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'support': support,
        'predicted': predicted,
    }

print('  PER-CLASS METRICS:')
print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10} {'Predicted':>10}")
print('  ' + '-' * 62)
for cls in ALERT_CLASSES:
    m = class_metrics[cls]
    print(f"  {cls:<10} {m['precision']:>9.1f}% {m['recall']:>9.1f}% {m['f1']:>9.1f}% {m['support']:>10} {m['predicted']:>10}")
print()

correct  = sum(confusion[c][c] for c in ALERT_CLASSES)
accuracy = correct / n_total * 100 if n_total > 0 else 0.0
classification_results = {
    'confusion_matrix': confusion,
    'class_metrics': class_metrics,
    'accuracy': round(accuracy, 1),
    'n_total': n_total,
}

# Lead time
print('='*70)
print(' LEAD TIME ANALYSIS (BASELINE SARIMA)')
print('='*70)

def compute_lead_time(df_a, target_class='RED'):
    df_a = df_a.copy()
    df_a['target_date'] = pd.to_datetime(df_a['target_date'])
    df_a['is_target']   = df_a['alert_actual'] == target_class

    events = []; curr = None
    for i, row in df_a.iterrows():
        if row['is_target']:
            if curr is None: curr = {'start_idx': i, 'start_date': row['target_date'], 'days': 1}
            else: curr['days'] += 1
        else:
            if curr: events.append(curr); curr = None
    if curr: events.append(curr)

    print(f'\n  {target_class} event blocks: {len(events)}')
    if not events: return None

    lead_times = []
    for event in events:
        si = event['start_idx']; lt = 0; found = False
        for j in range(si - 1, max(-1, si - 15), -1):
            st = df_a.loc[j, 'alert_status'] if j in df_a.index else 'GREEN'
            if st in ('YELLOW', 'RED'): lt = si - j; found = True
            else: break
        if not found:
            if df_a.loc[si, 'alert_status'] in ('YELLOW', 'RED'): lt, found = 0, True

        icon = '✓' if found else '✗'
        print(f'    {icon} {event["start_date"].date()} ({event["days"]}d {target_class}) → Lead: {lt}d' if found else f'    ✗ MISSED')
        lead_times.append({'detected': found, 'lead_time_days': lt if found else None})

    n_det   = sum(1 for lt in lead_times if lt['detected'])
    lt_vals = [lt['lead_time_days'] for lt in lead_times if lt['detected']]
    print(f'\n  Detected: {n_det}/{len(lead_times)}')
    if lt_vals:
        mean_lt = np.mean(lt_vals)
        print(f'  Mean lead time: {mean_lt:.1f} days  (target >= {TARGETS["Lead_Time_Avg"]} days)  '
              f'{"MET" if mean_lt >= TARGETS["Lead_Time_Avg"] else "NOT MET"}')

    result = {'target_class': target_class, 'n_events': len(lead_times),
              'n_detected': n_det, 'n_missed': len(lead_times) - n_det,
              'events': lead_times}
    if lt_vals:
        result.update({'mean_lead_time': round(np.mean(lt_vals), 1),
                       'min_lead_time': int(min(lt_vals)), 'max_lead_time': int(max(lt_vals))})
    return result


lead_time_results = compute_lead_time(df_alerts, 'RED')
if not lead_time_results or lead_time_results.get('n_events', 0) == 0:
    print('\n  No RED events — trying YELLOW...')
    lead_time_results = compute_lead_time(df_alerts, 'YELLOW')

mean_lead = lead_time_results.get('mean_lead_time', 0) if lead_time_results else 0

from vru_evaluation import run_full_evaluation
run_full_evaluation(forecast_results, df_alerts, naive_maes, DIR_PERF, TARGETS)

# =============================================================================
# HYPOTHESIS VALIDATION
# =============================================================================
print('\n' + '='*70)
print(' HYPOTHESIS VALIDATION (BASELINE SARIMA)')
print('='*70)

h1_met = avg_mape < TARGETS['MAPE_1day']
h2_met = accuracy >= 70.0
h3_met = mean_lead >= TARGETS['Lead_Time_Avg']

print(f'  H1 Forecast Accuracy   : {"   MET" if h1_met else "NOT MET"}  (avg MAPE={avg_mape:.1f}%)')
print(f'  H2 Alert Classification: {"   MET" if h2_met else "NOT MET"}  (accuracy={accuracy:.1f}%)')
print(f'  H3 Lead Time           : {"   MET" if h3_met else "NOT MET"}  (mean={mean_lead:.1f} days, target>={TARGETS["Lead_Time_Avg"]})')
print(f'\n  OVERALL: {sum([h1_met, h2_met, h3_met])}/3 hypotheses validated')
print('='*70)

# =============================================================================
# PLOTS
# =============================================================================
alert_colors = {'GREEN': '#22C55E', 'YELLOW': '#F59E0B', 'RED': '#EF4444', 'OFF': '#9CA3AF'}

# Forecast plot
fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
for ax, param in zip(axes, PARAMS):
    df_fc = forecast_results[param]
    dates = df_fc['target_date_5day'] if 'target_date_5day' in df_fc else df_fc['target_date']
    ax.plot(dates, df_fc['actual_5day'],   color='#1E40AF', linewidth=1, label='Actual at 5-day target')
    ax.plot(dates, df_fc['forecast_5day'], color='#93C5FD', linewidth=1, linestyle='--', label='SARIMA 5-day forecast')
    ax.fill_between(dates, df_fc['pi_lower_5day'], df_fc['pi_upper_5day'],
                    alpha=0.15, color='#3B82F6', label='95% PI')
    ax.set_ylabel(f"{PARAM_LABELS[param]}\n({PARAM_UNITS[param]})", fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
fig.suptitle('VRU Baseline SARIMA — Forecast vs Actual (Simple Threshold Alert)', fontsize=12)
plt.tight_layout()
plt.savefig(DIR_PLOTS + 'forecast_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()

# Alert timeline
fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True,
                          gridspec_kw={'height_ratios': [1, 1]})
ax = axes[0]
for i, row in df_alerts.iterrows():
    ax.barh(1, 1, left=i, color=alert_colors.get(row['alert_status'], 'gray'), height=0.4, align='center')
    ax.barh(0, 1, left=i, color=alert_colors.get(row['alert_actual'], 'gray'),  height=0.4, align='center')
ax.set_yticks([0, 1])
ax.set_yticklabels(['Ground Truth', 'System Alert'], fontsize=8)
patches = [mpatches.Patch(color=c, label=l) for l, c in alert_colors.items()]
ax.legend(handles=patches, loc='upper left', fontsize=7, ncol=4)
ax.set_title('Baseline SARIMA Alert Timeline (Simple Threshold, No CUSUM/EWMA)', fontsize=10)

ax2 = axes[1]
for param, color in zip(PARAMS, ['#3B82F6', '#F59E0B', '#EF4444']):
    df_fc = forecast_results[param]
    ax2.plot(range(len(df_fc)), df_fc['actual_1day'], linewidth=0.8,
             label=PARAM_LABELS[param], color=color)
ax2.legend(fontsize=7)
ax2.set_xlabel('Test Period (days)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(DIR_ALERTS + 'alert_timeline.png', dpi=150, bbox_inches='tight')
plt.close()

params = list(accuracy_results.keys())
labels = [p.replace('_', '\n').title() for p in params]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, metric, title in zip(axes, ['MAE', 'RMSE', 'MAPE', 'MASE'], ['MAE', 'RMSE', 'MAPE (%)', 'MASE']):
    v1 = [accuracy_results[p][f'{metric}_1day'] for p in params]
    v5 = [accuracy_results[p][f'{metric}_5day'] for p in params]
    x = np.arange(len(params))
    w = 0.35
    ax.bar(x - w/2, v1, w, label='1-Day', color='#3B82F6', alpha=0.8)
    ax.bar(x + w/2, v5, w, label='5-Day', color='#EF4444', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    if metric == 'MASE':
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
fig.suptitle('S2 SARIMA Forecast Accuracy by Parameter and Horizon', fontsize=12)
plt.tight_layout()
plt.savefig(DIR_PERF + 'forecast_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()

matrix = np.array([[confusion[fc].get(ac, 0) for ac in ALERT_CLASSES] for fc in ALERT_CLASSES])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(matrix, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(ALERT_CLASSES)))
ax.set_yticks(range(len(ALERT_CLASSES)))
ax.set_xticklabels(ALERT_CLASSES)
ax.set_yticklabels(ALERT_CLASSES)
ax.set_xlabel('Actual', fontsize=11)
ax.set_ylabel('Forecast', fontsize=11)
ax.set_title('S2 Alert Classification Confusion Matrix', fontsize=12)
matrix_max = matrix.max() if matrix.size else 0
for i in range(len(ALERT_CLASSES)):
    for j in range(len(ALERT_CLASSES)):
        val = matrix[i, j]
        color = 'white' if matrix_max > 0 and val > matrix_max * 0.6 else 'black'
        ax.text(j, i, str(val), ha='center', va='center', color=color, fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(DIR_PERF + 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nPlots saved.')

# =============================================================================
# SAVE RESULTS
# =============================================================================
performance_metrics = {
    'model':            'SARIMA + Box-Cox (baseline)',
    'alert_system':     'Simple threshold only (no CUSUM/EWMA/PI)',
    'split':            f'{int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}',
    'n_test':           n_total,
    'accuracy_results': accuracy_results,
    'forecast_accuracy': accuracy_results,
    'classification':    classification_results,
    'raw_fallback_counts': fallback_counts,
    'avg_mape_1day':    round(avg_mape, 2),
    'alert_accuracy':   round(accuracy, 1),
    'lead_time':        lead_time_results,
    'hypothesis': {
        'H1_met': h1_met, 'H2_met': h2_met, 'H3_met': h3_met,
        'overall': f'{sum([h1_met, h2_met, h3_met])}/3',
    },
}
with open(DIR_PERF + 'performance_metrics.json', 'w') as fh:
    json.dump(performance_metrics, fh, indent=2, default=str)

zip_path = OUT_ROOT + 'output_baseline_sarima.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(OUT_ROOT):
        for file in files:
            if file.endswith('.zip'): continue
            fp = os.path.join(root, file)
            zf.write(fp, os.path.relpath(fp, OUT_ROOT))
print(f'Zipped: {zip_path}')
print('\nDONE')
