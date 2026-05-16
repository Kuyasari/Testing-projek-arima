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

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import classification_report
from pmdarima import auto_arima

# =============================================================================
# PATHS
# =============================================================================

DATA_FILE  = 'vru_preprocessed.csv'
OUT_ROOT   = 'output_baseline_arima/'
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

CONFIRM = 1

TARGETS = {
    'MAPE_1day':    10.0,
    'MAPE_5day':    15.0,
    'MASE':          1.0,
    'Lead_Time_Avg': 3.0,
}

LEVEL = {'GREEN': 0, 'YELLOW': 1, 'RED': 2, 'OFF': -1}
REV   = {0: 'GREEN', 1: 'YELLOW', 2: 'RED', -1: 'OFF'}

print('=== BASELINE ARIMA SYSTEM ===')
print(f'  Data file : {DATA_FILE}')
print(f'  Split     : {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}')
print(f'  Horizon   : {HORIZON} days')
print(f'  Alert     : Simple threshold (no CUSUM/EWMA/PI)')



df_full = pd.read_csv(DATA_FILE, parse_dates=['date'])
df_full = df_full.sort_values('date').reset_index(drop=True)

print(f'\nFull dataset : {len(df_full)} rows')

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

print(f'Train : {len(train)} rows  ({train.date.min().date()} to {train.date.max().date()})')
print(f'Test  : {len(test)} rows   ({test.date.min().date()} to {test.date.max().date()})')


def detect_arima_order(series, param_name):
    model = auto_arima(series.dropna(), seasonal=False, stepwise=True,
                       information_criterion='aic', suppress_warnings=True,
                       error_action='ignore', max_p=5, max_q=5, max_d=2)
    o = model.order
    print(f'  {param_name}: auto_arima → ARIMA{o}  (AIC={model.aic():.1f})')
    return o


print('\nARIMA ORDER DETECTION')
print('─' * 50)
orders_all = {}
for param in PARAMS:
    print(f'\n  {PARAM_LABELS[param]}:')
    o = detect_arima_order(train[param], param)
    orders_all[param] = o

# Naive MAE baseline
naive_maes = {}
for param in PARAMS:
    series = df_valid[param].values
    naive_maes[param] = float(np.mean(np.abs(np.diff(series))))
    print(f'  Naive MAE [{param}]: {naive_maes[param]:.4f}')



print('\nStarting rolling walk-forward forecast...')
print(f'  Test points  : {len(test)}')

forecast_results = {}
n_test = len(test)

for param in PARAMS:
    print(f'\n--- {PARAM_LABELS[param]} ---')
    o = orders_all[param]

    records      = []
    fit          = None
    history_vals = list(train[param].values)

    for i in range(n_test - HORIZON):
        fc_date = test.iloc[i]['date']

        if fit is None or i % REFIT_INTERVAL == 0:
            try:
                hist_s = pd.Series(history_vals, dtype=float)
                model  = ARIMA(hist_s, order=o)
                fit    = model.fit()
                print(f'  [refit] step {i:3d}/{n_test - HORIZON}  ({fc_date.date()})  AIC={fit.aic:.1f}')
            except Exception as exc:
                print(f'  [refit FAILED step {i}]: {exc}')
                fit = None
                history_vals.append(float(test.iloc[i][param]))
                continue

        try:
            fc_obj   = fit.get_forecast(steps=HORIZON)
            fc_mean  = fc_obj.predicted_mean.values
            fc_ci    = fc_obj.conf_int(alpha=0.05)
            ci_lower = fc_ci.iloc[:, 0].values
            ci_upper = fc_ci.iloc[:, 1].values

            tgt_1d = test.iloc[i + 1]
            tgt_5d = test.iloc[i + HORIZON - 1]

            records.append({
                'forecast_date': fc_date,
                'target_date':   tgt_1d['date'],
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
        df_fc.to_csv(DIR_MODELS + f'forecast_{param}.csv', index=False)
        print(f'  {len(df_fc)} records -> {DIR_MODELS}forecast_{param}.csv')

    forecast_results[param] = df_fc

print('\nForecast done.')

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
n    = len(base)
raw_level_fc  = np.zeros(n, dtype=int)
raw_level_act = np.zeros(n, dtype=int)
alert_rows    = []

for i in range(n):
    row = {
        'forecast_date': base['forecast_date'].iloc[i],
        'target_date':   base['target_date'].iloc[i],
    }
    param_level_fc = []

    for param in PARAMS:
        df_fc  = forecast_results[param]
        actual = float(df_fc['actual_1day'].iloc[i])
        fc_5d  = float(df_fc['forecast_5day'].iloc[i])

        act_st    = classify_param_status(actual, param)
        fc_thr_st = classify_param_status(fc_5d,  param)

        # Baseline: hanya pakai 5-day forecast threshold (tidak ada CUSUM/EWMA/PI)
        lv = LEVEL.get(fc_thr_st, 0)
        param_level_fc.append(lv)

        row[f'{param}_act_status']  = act_st
        row[f'{param}_fc_status']   = REV[lv]
        row[f'{param}_actual']      = round(actual, 2)
        row[f'{param}_forecast_5d'] = round(fc_5d, 2)

        raw_level_act[i] = max(raw_level_act[i], LEVEL.get(act_st, 0))

    n_red    = sum(1 for lv in param_level_fc if lv >= 2)
    n_yellow = sum(1 for lv in param_level_fc if lv >= 1)

    if   n_red    >= 1: final_lv = 2
    elif n_yellow >= 1: final_lv = 1
    else:               final_lv = 0

    raw_level_fc[i] = final_lv
    alert_rows.append(row)

# Confirmation window
confirmed_fc = np.zeros(n, dtype=int)
for i in range(n):
    win = raw_level_fc[max(0, i - CONFIRM + 1) : i + 1]
    if len(win) >= 1 and win.max() >= 2: confirmed_fc[i] = 2
    elif len(win) >= 1 and win.max() >= 1: confirmed_fc[i] = 1
    else: confirmed_fc[i] = 0

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


def safe_mape(actual, forecast):
    a, f = np.array(actual, dtype=float), np.array(forecast, dtype=float)
    mask = np.abs(a) > 1e-6
    if mask.sum() == 0:
        return np.nan, 0
    return np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100, int((~mask).sum())

accuracy_results = {}
mape_all = []

print('\n' + '='*70)
print(' FORECAST ACCURACY (BASELINE ARIMA)')
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

    mape_all.append(mape_1d)

    print(f'\n  {label} ({unit})')
    print(f'  {"─"*38}')
    print(f'  {"Metric":<8} {"1-Day":>8} {"5-Day":>8}')
    print(f'  {"MAE":<8} {mae_1d:>8.3f} {mae_5d:>8.3f}  {unit}')
    print(f'  {"RMSE":<8} {rmse_1d:>8.3f} {rmse_5d:>8.3f}  {unit}')
    print(f'  {"MAPE":<8} {mape_1d:>7.2f}% {mape_5d:>7.2f}%')
    print(f'  {"MASE":<8} {mase_1d:>8.3f}  ({"< 1.0 beats naive" if mase_1d < 1 else "> 1.0 below naive"})')

    accuracy_results[param] = {
        'MAE_1day':  round(mae_1d, 4),  'MAE_5day':  round(mae_5d, 4),
        'RMSE_1day': round(rmse_1d, 4), 'RMSE_5day': round(rmse_5d, 4),
        'MAPE_1day': round(mape_1d, 2), 'MAPE_5day': round(mape_5d, 2),
        'MASE_1day': round(mase_1d, 3),
        'naive_mae': round(naive_mae, 4),
        'n_forecasts': len(df_fc),
    }

avg_mape = float(np.mean(mape_all))

# Alert classification
ALERT_CLASSES = ['OFF', 'GREEN', 'YELLOW', 'RED']
y_pred  = df_alerts['alert_status'].values
y_true  = df_alerts['alert_actual'].values
n_total = len(df_alerts)

print('\n' + '='*70)
print(' ALERT CLASSIFICATION PERFORMANCE (BASELINE)')
print('='*70)
print(f'\n  Total test points: {n_total}')
print()
print(classification_report(y_true, y_pred, labels=['GREEN', 'YELLOW', 'RED'], zero_division=0))

correct  = sum((y_pred == y_true))
accuracy = correct / n_total * 100

# Lead time
print('='*70)
print(' LEAD TIME ANALYSIS (BASELINE)')
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
        print(f'    {icon} {event["start_date"].date()} ({event["days"]}d {target_class}) → Lead time: {lt}d' if found else f'    ✗ MISSED')
        lead_times.append({'detected': found, 'lead_time_days': lt if found else None})

    n_det   = sum(1 for lt in lead_times if lt['detected'])
    lt_vals = [lt['lead_time_days'] for lt in lead_times if lt['detected']]
    print(f'\n  Detected: {n_det}/{len(lead_times)}')
    if lt_vals:
        mean_lt = np.mean(lt_vals)
        print(f'  Mean lead time: {mean_lt:.1f} days  (target >= {TARGETS["Lead_Time_Avg"]} days)  '
              f'{"MET" if mean_lt >= TARGETS["Lead_Time_Avg"] else "NOT MET"}')

    result = {'target_class': target_class, 'n_events': len(lead_times),
              'n_detected': n_det, 'events': lead_times}
    if lt_vals:
        result.update({'mean_lead_time': round(np.mean(lt_vals), 1),
                       'min_lead_time': int(min(lt_vals)), 'max_lead_time': int(max(lt_vals))})
    return result


lead_time_results = compute_lead_time(df_alerts, 'RED')
if not lead_time_results or lead_time_results.get('n_events', 0) == 0:
    print('\n  No RED events — trying YELLOW...')
    lead_time_results = compute_lead_time(df_alerts, 'YELLOW')

mean_lead = lead_time_results.get('mean_lead_time', 0) if lead_time_results else 0


print('\n' + '='*70)
print(' HYPOTHESIS VALIDATION (BASELINE ARIMA)')
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

colors = {'running': '#3B82F6', 'degrading': '#F59E0B', 'down': '#EF4444'}
alert_colors = {'GREEN': '#22C55E', 'YELLOW': '#F59E0B', 'RED': '#EF4444', 'OFF': '#9CA3AF'}

# Forecast plot
fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
for ax, param in zip(axes, PARAMS):
    df_fc = forecast_results[param]
    dates = df_fc['target_date']
    ax.plot(dates, df_fc['actual_1day'],   color='#1E40AF', linewidth=1, label='Actual')
    ax.plot(dates, df_fc['forecast_5day'], color='#93C5FD', linewidth=1, linestyle='--', label='ARIMA 5-day forecast')
    ax.fill_between(dates, df_fc['pi_lower_5day'], df_fc['pi_upper_5day'],
                    alpha=0.15, color='#3B82F6', label='95% PI')
    ax.set_ylabel(f"{PARAM_LABELS[param]}\n({PARAM_UNITS[param]})", fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
fig.suptitle('VRU Baseline — ARIMA Forecast vs Actual', fontsize=12)
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
ax.set_title('Baseline Alert Timeline (Simple Threshold)', fontsize=10)

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

print('\nPlots saved.')
from vru_evaluation import run_full_evaluation
run_full_evaluation(forecast_results, df_alerts, naive_maes, DIR_PERF, TARGETS)

performance_metrics = {
    'model':           'ARIMA (baseline)',
    'alert_system':    'Simple threshold only',
    'split':           f'{int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}',
    'n_test':          n_total,
    'accuracy_results': accuracy_results,
    'avg_mape_1day':   round(avg_mape, 2),
    'alert_accuracy':  round(accuracy, 1),
    'lead_time':       lead_time_results,
    'hypothesis': {
        'H1_met': h1_met, 'H2_met': h2_met, 'H3_met': h3_met,
        'overall': f'{sum([h1_met, h2_met, h3_met])}/3',
    },
}
with open(DIR_PERF + 'performance_metrics.json', 'w') as fh:
    json.dump(performance_metrics, fh, indent=2, default=str)

# Zip output
zip_path = OUT_ROOT + 'output_baseline_arima.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(OUT_ROOT):
        for file in files:
            if file.endswith('.zip'): continue
            fp = os.path.join(root, file)
            zf.write(fp, os.path.relpath(fp, OUT_ROOT))
print(f'Zipped: {zip_path}')

print('\nDONE')
