
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, json, os, zipfile
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import classification_report
from pmdarima import auto_arima

# ── PATHS ──
DATA_FILE  = 'vru_preprocessed.csv'
OUT_ROOT   = 'output_arima_enhanced/'
DIR_MODELS = OUT_ROOT + 'arima_models/'
DIR_ALERTS = OUT_ROOT + 'alerts/'
DIR_PLOTS  = OUT_ROOT + 'plots/'
DIR_PERF   = OUT_ROOT + 'performance_results/'
for d in [DIR_MODELS, DIR_ALERTS, DIR_PLOTS, DIR_PERF]:
    os.makedirs(d, exist_ok=True)

# ── CONFIG ──
PARAMS       = ['discharge_temp', 'discharge_pressure', 'jacket_water']
PARAM_UNITS  = {'discharge_temp': '°F', 'discharge_pressure': 'psi', 'jacket_water': 'psi'}
PARAM_LABELS = {'discharge_temp': 'Discharge Temperature',
                'discharge_pressure': 'Discharge Pressure',
                'jacket_water': 'Jacket Water Pressure'}

TRAIN_RATIO    = 0.80
HORIZON        = 5
REFIT_INTERVAL = 20

CUSUM_K_FACTOR = 0.5
CUSUM_H_FACTOR = 4.0
EWMA_LAMBDA    = 0.3
EWMA_L         = 2.5
PI_WINDOW      = 7

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

CONFIRM = 1
TARGETS = {'MAPE_1day': 10.0, 'MAPE_5day': 15.0, 'MASE': 1.0, 'Lead_Time_Avg': 3.0}
LEVEL   = {'GREEN': 0, 'YELLOW': 1, 'RED': 2, 'OFF': -1}
REV     = {0: 'GREEN', 1: 'YELLOW', 2: 'RED', -1: 'OFF'}

print('=== ARIMA (auto_arima) + CUSUM + EWMA + PI SYSTEM ===')
print(f'  Forecast : ARIMA with auto_arima order selection (no seasonal, no Box-Cox)')
print(f'  Alert    : CUSUM + EWMA + PI')

# ── LOAD & PREPROCESS ──
df_full = pd.read_csv(DATA_FILE, parse_dates=['date']).sort_values('date').reset_index(drop=True)
df = df_full.copy().reset_index(drop=True).set_index('date')  # down-days already removed in preprocessing
full_idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
df = df.reindex(full_idx)
for col in PARAMS:
    df[col] = df[col].ffill(limit=3)
df = df.reset_index().rename(columns={'index': 'date'})
df_valid = df.dropna(subset=PARAMS).reset_index(drop=True)
split_idx = int(len(df_valid) * TRAIN_RATIO)
train = df_valid.iloc[:split_idx].copy()
test  = df_valid.iloc[split_idx:].copy().reset_index(drop=True)
print(f'  Train: {len(train)} | Test: {len(test)}')

# ── ARIMA ORDER (auto_arima, non-seasonal) ──
def detect_arima_order(series, param_name):
    model = auto_arima(series.dropna(), seasonal=False, stepwise=True,
                       information_criterion='aic', suppress_warnings=True,
                       error_action='ignore', max_p=5, max_q=5, max_d=2)
    o = model.order
    print(f'  {param_name}: auto_arima → ARIMA{o}  (AIC={model.aic():.1f})')
    return o

orders_all = {p: detect_arima_order(train[p], p) for p in PARAMS}
naive_maes = {p: float(np.mean(np.abs(np.diff(df_valid[p].values)))) for p in PARAMS}

# ── ROLLING FORECAST ──
print('\nRolling forecast...')
forecast_results = {}
n_test = len(test)

for param in PARAMS:
    print(f'\n--- {PARAM_LABELS[param]} ---')
    o = orders_all[param]
    records, fit, history_vals = [], None, list(train[param].values)

    for i in range(n_test - HORIZON):
        fc_date = test.iloc[i]['date']
        if fit is None or i % REFIT_INTERVAL == 0:
            try:
                fit = ARIMA(pd.Series(history_vals, dtype=float), order=o).fit()
                print(f'  [refit] step {i:3d}/{n_test-HORIZON}  ({fc_date.date()})  AIC={fit.aic:.1f}')
            except Exception as exc:
                print(f'  [refit FAILED {i}]: {exc}'); fit = None
                history_vals.append(float(test.iloc[i][param])); continue
        try:
            fc_obj = fit.get_forecast(steps=HORIZON)
            fc_mean, fc_ci = fc_obj.predicted_mean.values, fc_obj.conf_int(alpha=0.05)
            tgt_1d, tgt_5d = test.iloc[i+1], test.iloc[i+HORIZON-1]
            records.append({
                'forecast_date': fc_date, 'target_date': tgt_1d['date'],
                'target_date_1day': tgt_1d['date'], 'target_date_5day': tgt_5d['date'],
                'actual_1day': float(tgt_1d[param]), 'forecast_1day': float(fc_mean[0]),
                'pi_lower_1day': float(fc_ci.iloc[0,0]), 'pi_upper_1day': float(fc_ci.iloc[0,1]),
                'actual_5day': float(tgt_5d[param]), 'forecast_5day': float(fc_mean[HORIZON-1]),
                'pi_lower_5day': float(fc_ci.iloc[HORIZON-1,0]), 'pi_upper_5day': float(fc_ci.iloc[HORIZON-1,1]),
            })
        except: pass
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

# Dedicated 5-day forecast chart. The alert timeline below also has forecast
# lines, but this plot aligns the 5-day forecast with the 5-day target date.
fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
for ax, param in zip(axes, PARAMS):
    df_fc = forecast_results[param]
    dates_5d = df_fc['target_date_5day'] if 'target_date_5day' in df_fc else df_fc['target_date']
    ax.plot(dates_5d, df_fc['actual_5day'], color='#1E40AF', linewidth=1.1, label='Actual at 5-day target')
    ax.plot(dates_5d, df_fc['forecast_5day'], color='#EF4444', linewidth=1.1, linestyle='--', label='ARIMA 5-day forecast')
    ax.fill_between(dates_5d, df_fc['pi_lower_5day'], df_fc['pi_upper_5day'],
                    alpha=0.14, color='#EF4444', label='95% PI (5-day)')
    ax.set_ylabel(f"{PARAM_LABELS[param]}\n({PARAM_UNITS[param]})", fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
fig.suptitle('S3 ARIMA + CUSUM + EWMA - 5-Day Forecast vs Actual', fontsize=12)
plt.tight_layout()
plt.savefig(DIR_PLOTS + 'forecast_vs_actual_5day.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"5-day forecast plot saved: {DIR_PLOTS}forecast_vs_actual_5day.png")

# ── CUSUM + EWMA + PI DETECTORS ──
def compute_cusum(residuals, k_factor=0.5, h_factor=4.0):
    res = np.array(residuals, dtype=float)
    mean_r, std_r = np.nanmean(res), np.nanstd(res)
    k, h = k_factor * std_r, h_factor * std_r
    cp, cn, alert = np.zeros(len(res)), np.zeros(len(res)), np.zeros(len(res), dtype=bool)
    for i in range(1, len(res)):
        if np.isnan(res[i]): cp[i]=cp[i-1]; cn[i]=cn[i-1]; continue
        cp[i] = max(0, cp[i-1] + (res[i]-mean_r) - k)
        cn[i] = max(0, cn[i-1] - (res[i]-mean_r) - k)
        if cp[i] > h or cn[i] > h:
            alert[i] = True; cp[i] = 0; cn[i] = 0
    return pd.DataFrame({'cusum_pos': cp, 'cusum_neg': cn, 'cusum_alert': alert,
                         'cusum_h': h, 'residual': res})

def compute_ewma(residuals, lam=0.3, L=2.5):
    res = np.array(residuals, dtype=float)
    mean_r, std_r = np.nanmean(res), np.nanstd(res)
    sigma_ewma = std_r * np.sqrt(lam / (2 - lam))
    ucl, lcl = mean_r + L * sigma_ewma, mean_r - L * sigma_ewma
    z, alert = np.zeros(len(res)), np.zeros(len(res), dtype=bool)
    z[0] = mean_r
    for i in range(1, len(res)):
        if np.isnan(res[i]): z[i] = z[i-1]; continue
        z[i] = lam * res[i] + (1 - lam) * z[i-1]
        if z[i] > ucl or z[i] < lcl: alert[i] = True
    return pd.DataFrame({'ewma_z': z, 'ewma_alert': alert, 'ewma_ucl': ucl, 'ewma_lcl': lcl})

def compute_pi_violation_rate(actuals, uppers, lowers, window=7):
    violated = (np.array(actuals) > np.array(uppers)) | (np.array(actuals) < np.array(lowers))
    rate = np.full(len(actuals), np.nan)
    for i in range(window-1, len(actuals)): rate[i] = violated[i-window+1:i+1].mean()
    return rate, violated

def classify_param_status(value, param):
    thr = THRESHOLDS[param]
    if param == 'discharge_temp':
        if value < thr['OFF']: return 'OFF'
        if value >= thr['RED']: return 'RED'
        if value >= thr['YELLOW']: return 'YELLOW'
        return 'GREEN'
    else:
        if value < thr['LOW'] or value > thr['HIGH']: return 'YELLOW'
        return 'GREEN'

# Compute CUSUM, EWMA, PI for each parameter
base = forecast_results[PARAMS[0]][['forecast_date', 'target_date']].copy()
cusum_results, ewma_results, pi_results = {}, {}, {}
for param in PARAMS:
    df_fc = forecast_results[param]
    if df_fc.empty: continue
    res = df_fc['actual_1day'].values - df_fc['forecast_1day'].values
    cusum_results[param] = compute_cusum(res, CUSUM_K_FACTOR, CUSUM_H_FACTOR)
    ewma_results[param]  = compute_ewma(res, EWMA_LAMBDA, EWMA_L)
    rate, violated = compute_pi_violation_rate(
        df_fc['actual_1day'].values, df_fc['pi_upper_1day'].values,
        df_fc['pi_lower_1day'].values, PI_WINDOW)
    pi_results[param] = {'rate': rate, 'violated': violated}

# ── BUILD ALERT TABLE ──
n = len(base)
raw_level_fc  = np.zeros(n, dtype=int)
raw_level_act = np.zeros(n, dtype=int)
alert_rows = []

for i in range(n):
    row = {'forecast_date': base['forecast_date'].iloc[i], 'target_date': base['target_date'].iloc[i]}
    reasons, param_level_fc = [], []

    for param in PARAMS:
        df_fc = forecast_results[param]
        actual, fc_5d = float(df_fc['actual_1day'].iloc[i]), float(df_fc['forecast_5day'].iloc[i])
        act_st    = classify_param_status(actual, param)
        fc_thr_st = classify_param_status(fc_5d, param)
        cusum_alert = bool(cusum_results[param]['cusum_alert'].iloc[i])
        ewma_alert  = bool(ewma_results[param]['ewma_alert'].iloc[i])
        pi_rate = pi_results[param]['rate'][i]
        thr_p = PI_THRESHOLDS[param]
        pi_st = 'GREEN'
        if not np.isnan(pi_rate):
            if pi_rate >= thr_p['red']: pi_st = 'RED'
            elif pi_rate >= thr_p['yellow']: pi_st = 'YELLOW'

        lv = max(LEVEL.get(fc_thr_st, 0), LEVEL.get(pi_st, 0),
                 1 if cusum_alert else 0, 1 if ewma_alert else 0)
        param_level_fc.append(lv)

        parts = []
        if cusum_alert: parts.append(f'{param}:CUSUM')
        if ewma_alert:  parts.append(f'{param}:EWMA')
        if pi_st != 'GREEN': parts.append(f'{param}:PI={pi_rate:.2f}({pi_st})')
        reasons.extend(parts)

        row[f'{param}_fc_status']   = REV[lv]
        row[f'{param}_act_status']  = act_st
        row[f'{param}_actual']      = round(actual, 2)
        row[f'{param}_forecast_5d'] = round(fc_5d, 2)
        raw_level_act[i] = max(raw_level_act[i], LEVEL.get(act_st, 0))

    n_red    = sum(1 for lv in param_level_fc if lv >= 2)
    n_yellow = sum(1 for lv in param_level_fc if lv >= 1)
    if n_red >= 1: final_lv = 2
    elif n_yellow >= 1: final_lv = 1
    else: final_lv = 0
    raw_level_fc[i] = final_lv
    row['alert_reason'] = '; '.join(reasons) if reasons else 'Normal'
    alert_rows.append(row)

confirmed_fc = np.zeros(n, dtype=int)
for i in range(n):
    win = raw_level_fc[max(0, i-CONFIRM+1):i+1]
    if win.max() >= 2: confirmed_fc[i] = 2
    elif win.max() >= 1: confirmed_fc[i] = 1

for i in range(n):
    alert_rows[i]['alert_status'] = REV[confirmed_fc[i]]
    alert_rows[i]['alert_actual'] = REV[raw_level_act[i]]

df_alerts = pd.DataFrame(alert_rows)
df_alerts.to_csv(DIR_ALERTS + 'alerts_generated.csv', index=False)
print(f'\nAlerts: {len(df_alerts)} rows')
print(df_alerts['alert_status'].value_counts().to_string())

# ── PERFORMANCE ──
def safe_mape(a, f):
    a, f = np.array(a, dtype=float), np.array(f, dtype=float)
    mask = np.abs(a) > 1e-6
    return (np.mean(np.abs((a[mask]-f[mask])/a[mask]))*100 if mask.sum() else np.nan)

mape_all = []
print('\n' + '='*70 + '\n FORECAST ACCURACY (ARIMA + CUSUM + EWMA)\n' + '='*70)
for param in PARAMS:
    df_fc = forecast_results[param]
    a1, f1 = df_fc['actual_1day'].values, df_fc['forecast_1day'].values
    a5, f5 = df_fc['actual_5day'].values, df_fc['forecast_5day'].values
    mae_1d, rmse_1d = np.mean(np.abs(a1-f1)), np.sqrt(np.mean((a1-f1)**2))
    mape_1d, mape_5d = safe_mape(a1, f1), safe_mape(a5, f5)
    mase_1d = mae_1d / naive_maes[param]
    mape_all.append(mape_1d)
    print(f'\n  {PARAM_LABELS[param]}: MAPE={mape_1d:.2f}%/{mape_5d:.2f}%  MASE={mase_1d:.3f}')

avg_mape = np.mean(mape_all)

y_pred, y_true = df_alerts['alert_status'].values, df_alerts['alert_actual'].values
n_total = len(df_alerts)
print('\n' + '='*70 + '\n ALERT CLASSIFICATION\n' + '='*70)
print(classification_report(y_true, y_pred, labels=['GREEN','YELLOW','RED'], zero_division=0))
accuracy = (y_pred == y_true).sum() / n_total * 100

# ── LEAD TIME ──
print('='*70 + '\n LEAD TIME ANALYSIS\n' + '='*70)
def compute_lead_time(df_a, target_class='RED'):
    df_a = df_a.copy()
    df_a['target_date'] = pd.to_datetime(df_a['target_date'])
    events, curr = [], None
    for i, row in df_a.iterrows():
        if row['alert_actual'] == target_class:
            if curr is None: curr = {'start_idx': i, 'start_date': row['target_date'], 'days': 1}
            else: curr['days'] += 1
        else:
            if curr: events.append(curr); curr = None
    if curr: events.append(curr)
    print(f'\n  {target_class} event blocks: {len(events)}')
    if not events: return None
    lead_times = []
    for ev in events:
        si, lt, found = ev['start_idx'], 0, False
        for j in range(si-1, max(-1, si-15), -1):
            st = df_a.loc[j, 'alert_status'] if j in df_a.index else 'GREEN'
            if st in ('YELLOW', 'RED'): lt = si - j; found = True
            else: break
        if not found and df_a.loc[si, 'alert_status'] in ('YELLOW', 'RED'): lt, found = 0, True
        icon = '[v]' if found else '[x]'
        lt_s = f'{lt}d' if found else 'MISSED'
        print(f'    {icon} {ev["start_date"].date()} ({ev["days"]}d) -> Lead: {lt_s}')
        lead_times.append({'detected': found, 'lead_time_days': lt if found else None})
    lt_vals = [x['lead_time_days'] for x in lead_times if x['detected']]
    if lt_vals:
        mean_lt = np.mean(lt_vals)
        print(f'\n  Mean lead time: {mean_lt:.1f} days  (target >= 3.0)  {"MET" if mean_lt >= 3 else "NOT MET"}')
    return {'mean_lead_time': round(np.mean(lt_vals), 1) if lt_vals else 0,
            'n_events': len(events), 'n_detected': sum(1 for x in lead_times if x['detected'])}

lt_results = compute_lead_time(df_alerts, 'RED')
if not lt_results or lt_results.get('n_events', 0) == 0:
    lt_results = compute_lead_time(df_alerts, 'YELLOW')
mean_lead = lt_results.get('mean_lead_time', 0) if lt_results else 0

from vru_evaluation import run_full_evaluation
run_full_evaluation(forecast_results, df_alerts, naive_maes, DIR_PERF, TARGETS)

# ── HYPOTHESIS ──
h1 = avg_mape < TARGETS['MAPE_1day']
h2 = accuracy >= 70.0
h3 = mean_lead >= TARGETS['Lead_Time_Avg']
print('\n' + '='*70)
print(' HYPOTHESIS VALIDATION (ARIMA + CUSUM + EWMA)')
print('='*70)
print(f'  H1 Forecast Accuracy   : {"   MET" if h1 else "NOT MET"}  (avg MAPE={avg_mape:.1f}%)')
print(f'  H2 Alert Classification: {"   MET" if h2 else "NOT MET"}  (accuracy={accuracy:.1f}%)')
print(f'  H3 Lead Time           : {"   MET" if h3 else "NOT MET"}  (mean={mean_lead:.1f} days)')
print(f'\n  OVERALL: {sum([h1,h2,h3])}/3 hypotheses validated')
print('='*70)


# ── SAVE ──
json.dump({'model': 'ARIMA + CUSUM + EWMA + PI', 'avg_mape': round(avg_mape,2),
           'accuracy': round(accuracy,1), 'lead_time': mean_lead,
           'hypothesis': {'H1': h1, 'H2': h2, 'H3': h3, 'overall': f'{sum([h1,h2,h3])}/3'}},
          open(DIR_PERF + 'performance_metrics.json', 'w'), indent=2, default=str)

# Alert timeline plot
alert_colors = {'GREEN': '#22C55E', 'YELLOW': '#F59E0B', 'RED': '#EF4444', 'OFF': '#9CA3AF'}
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [2,2,2,1]})
for ax, param in zip(axes[:3], PARAMS):
    df_fc = forecast_results[param]
    dates = df_fc['target_date_5day'] if 'target_date_5day' in df_fc else df_fc['target_date']
    ax.plot(dates, df_fc['actual_5day'], color='#1E40AF', linewidth=1, label='Actual at 5-day target')
    ax.plot(dates, df_fc['forecast_5day'], color='#93C5FD', linewidth=1, linestyle='--', label='ARIMA 5d fc')
    ax.fill_between(dates, df_fc['pi_lower_5day'], df_fc['pi_upper_5day'], alpha=0.15, color='#3B82F6')
    ax.set_ylabel(f"{PARAM_LABELS[param]}\n({PARAM_UNITS[param]})", fontsize=8)
    ax.legend(loc='upper left', fontsize=7); ax.grid(True, alpha=0.3)
ax = axes[3]
for i, row in df_alerts.iterrows():
    ax.barh(1, 1, left=i, color=alert_colors.get(row['alert_status'], 'gray'), height=0.4, align='center')
    ax.barh(0, 1, left=i, color=alert_colors.get(row['alert_actual'], 'gray'), height=0.4, align='center')
ax.set_yticks([0,1]); ax.set_yticklabels(['Ground Truth', 'System Alert'], fontsize=8)
patches = [mpatches.Patch(color=c, label=l) for l, c in alert_colors.items()]
ax.legend(handles=patches, fontsize=7, ncol=4)
fig.suptitle('ARIMA + CUSUM + EWMA Alert System', fontsize=12)
plt.tight_layout()
plt.savefig(DIR_PLOTS + 'alert_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nPlots saved.\nDONE')
