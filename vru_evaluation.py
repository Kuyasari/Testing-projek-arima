"""
vru_evaluation.py — Shared evaluation module for S1/S2/S3/S4

Import this in any system script to get:
  - classify_param_status()   → classify one parameter value
  - classify_system_status()  → classify system from 3 param values
  - compute_full_metrics()    → MAE/RMSE/MAPE/MASE × 2 horizons
  - compute_alert_metrics()   → confusion matrix + P/R/F1 per class
  - compute_lead_time()       → backward-search lead time analysis
  - run_full_evaluation()     → runs everything, saves JSON + plots

Usage in any S1-S4 script:
  from vru_evaluation import run_full_evaluation
  run_full_evaluation(forecast_results, df_alerts, naive_maes, output_dir, targets)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

# =============================================================================
# THRESHOLDS — single source of truth for all 4 systems
# =============================================================================

THRESHOLDS = {
    'discharge_temp':     {'OFF': 90, 'YELLOW': 150, 'RED': 160, 'CRITICAL': 300},
    'discharge_pressure': {'LOW': 10, 'HIGH': 30},
    'jacket_water':       {'LOW': 12, 'HIGH': 20},
}

PARAMS = ['discharge_temp', 'discharge_pressure', 'jacket_water']
PARAM_LABELS = {
    'discharge_temp': 'Discharge Temperature',
    'discharge_pressure': 'Discharge Pressure',
    'jacket_water': 'Jacket Water Pressure',
}
PARAM_UNITS = {'discharge_temp': '°F', 'discharge_pressure': 'psi', 'jacket_water': 'psi'}
ALERT_CLASSES = ['OFF', 'GREEN', 'YELLOW', 'RED']
LEVEL = {'GREEN': 0, 'YELLOW': 1, 'RED': 2, 'OFF': -1}
REV   = {0: 'GREEN', 1: 'YELLOW', 2: 'RED', -1: 'OFF'}


# =============================================================================
# CLASSIFICATION — matches all 4 system codes
# =============================================================================

def classify_param_status(value, param):
    """Classify a single parameter. Returns 'OFF'/'GREEN'/'YELLOW'/'RED'."""
    thr = THRESHOLDS[param]
    if param == 'discharge_temp':
        if value < thr['OFF']:      return 'OFF'
        if value >= thr['CRITICAL']: return 'RED'
        if value >= thr['RED']:     return 'RED'
        if value >= thr['YELLOW']:  return 'YELLOW'
        return 'GREEN'
    else:
        if value < thr['LOW'] or value > thr['HIGH']:
            return 'YELLOW'
        return 'GREEN'


def classify_system_status(temp_val, press_val, jacket_val):
    """
    System-level alert from 3 parameter values.
    Voting: 1 param RED → system RED, 1 param YELLOW → system YELLOW.
    Returns dict with status, reason, details.
    """
    if temp_val < THRESHOLDS['discharge_temp']['OFF']:
        return {'status': 'OFF', 'reason': f'Compressor OFF ({temp_val:.1f}°F)',
                'details': {'discharge_temp': 'OFF', 'discharge_pressure': 'N/A', 'jacket_water': 'N/A'}}

    statuses = {
        'discharge_temp': classify_param_status(temp_val, 'discharge_temp'),
        'discharge_pressure': classify_param_status(press_val, 'discharge_pressure'),
        'jacket_water': classify_param_status(jacket_val, 'jacket_water'),
    }
    max_lv = max(LEVEL.get(s, 0) for s in statuses.values())
    final = REV.get(max_lv, 'GREEN')

    parts = []
    vals = {'discharge_temp': temp_val, 'discharge_pressure': press_val, 'jacket_water': jacket_val}
    for p, s in statuses.items():
        if s in ('YELLOW', 'RED'):
            parts.append(f'{PARAM_LABELS[p]}: {vals[p]:.1f} → {s}')

    return {'status': final, 'reason': '; '.join(parts) if parts else 'Normal',
            'details': statuses}


# =============================================================================
# FORECAST ACCURACY — MAE / RMSE / MAPE / MASE × 2 horizons
# =============================================================================

def safe_mape(a, f):
    a, f = np.array(a, dtype=float), np.array(f, dtype=float)
    mask = np.abs(a) > 1e-6
    if mask.sum() == 0: return np.nan
    return float(np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100)


def compute_full_metrics(forecast_results, naive_maes):
    """
    Compute MAE/RMSE/MAPE/MASE for 1-day and 5-day horizons.

    Args:
        forecast_results: dict of {param: DataFrame with actual_1day, forecast_1day, actual_5day, forecast_5day}
        naive_maes: dict of {param: float}

    Returns:
        dict of {param: {MAE_1day, MAE_5day, RMSE_1day, RMSE_5day, MAPE_1day, MAPE_5day, MASE_1day, MASE_5day, ...}}
    """
    results = {}

    for param in PARAMS:
        if param not in forecast_results:
            continue
        df = forecast_results[param]
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            a1, f1 = df['actual_1day'].values, df['forecast_1day'].values
            a5, f5 = df['actual_5day'].values, df['forecast_5day'].values
        else:
            continue

        mae_1d  = float(np.mean(np.abs(a1 - f1)))
        mae_5d  = float(np.mean(np.abs(a5 - f5)))
        rmse_1d = float(np.sqrt(np.mean((a1 - f1)**2)))
        rmse_5d = float(np.sqrt(np.mean((a5 - f5)**2)))
        mape_1d = safe_mape(a1, f1)
        mape_5d = safe_mape(a5, f5)

        nm = naive_maes.get(param, 1.0)
        if nm == 0: nm = 1.0
        mase_1d = mae_1d / nm
        mase_5d = mae_5d / nm

        results[param] = {
            'MAE_1day': round(mae_1d, 4), 'MAE_5day': round(mae_5d, 4),
            'RMSE_1day': round(rmse_1d, 4), 'RMSE_5day': round(rmse_5d, 4),
            'MAPE_1day': round(mape_1d, 2) if mape_1d is not None and not np.isnan(mape_1d) else float('nan'),
            'MAPE_5day': round(mape_5d, 2) if mape_5d is not None and not np.isnan(mape_5d) else float('nan'),
            'MASE_1day': round(mase_1d, 3), 'MASE_5day': round(mase_5d, 3),
            'naive_mae': round(nm, 4), 'n_forecasts': len(df),
        }

    return results


# =============================================================================
# ALERT CLASSIFICATION METRICS — confusion + P/R/F1 per class
# =============================================================================

def compute_alert_metrics(df_alerts):
    """
    Compute confusion matrix, per-class P/R/F1, and overall accuracy.

    Args:
        df_alerts: DataFrame with 'alert_status' (forecast) and 'alert_actual' (ground truth)

    Returns:
        dict with confusion_matrix, class_metrics, accuracy, n_total
    """
    yp = df_alerts['alert_status'].values
    yt = df_alerts['alert_actual'].values
    n = len(df_alerts)

    # Confusion matrix
    cm = {}
    for fc in ALERT_CLASSES:
        cm[fc] = {}
        for ac in ALERT_CLASSES:
            cm[fc][ac] = int(((yp == fc) & (yt == ac)).sum())

    # Per-class metrics
    met = {}
    for cls in ALERT_CLASSES:
        tp = cm[cls].get(cls, 0)
        fp = sum(cm[cls].get(o, 0) for o in ALERT_CLASSES if o != cls)
        fn = sum(cm[f].get(cls, 0) for f in ALERT_CLASSES if f != cls)
        sup = tp + fn; pred = tp + fp
        pr = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
        met[cls] = {'precision': round(pr, 1), 'recall': round(rc, 1), 'f1': round(f1, 1),
                    'tp': tp, 'fp': fp, 'fn': fn, 'support': sup, 'predicted': pred}

    correct = sum(cm[c].get(c, 0) for c in ALERT_CLASSES)
    accuracy = correct / n * 100 if n > 0 else 0

    return {'confusion_matrix': cm, 'class_metrics': met, 'accuracy': round(accuracy, 1), 'n_total': n}


# =============================================================================
# LEAD TIME — backward search
# =============================================================================

def compute_lead_time(df_alerts, target_class='RED', fallback='YELLOW'):
    """
    Compute lead time: days before actual RED, forecast showed YELLOW/RED.

    Returns:
        dict with n_events, n_detected, n_missed, detection_rate, mean/min/max lead time
    """
    df = df_alerts.copy()
    df['target_date'] = pd.to_datetime(df['target_date'])
    yp = df['alert_status'].values
    yt = df['alert_actual'].values
    n = len(df)

    # Find event blocks
    def find_events(target):
        events = []; curr = None
        for i in range(n):
            if yt[i] == target:
                if curr is None: curr = {'s': i, 'e': i, 'date': df.iloc[i]['target_date']}
                else: curr['e'] = i
            else:
                if curr: curr['d'] = curr['e'] - curr['s'] + 1; events.append(curr); curr = None
        if curr: curr['d'] = curr['e'] - curr['s'] + 1; events.append(curr)
        return events

    events = find_events(target_class)
    used_class = target_class
    if not events and fallback:
        events = find_events(fallback)
        used_class = fallback

    if not events:
        return {'target_class': used_class, 'n_events': 0, 'n_detected': 0, 'n_missed': 0, 'detection_rate': 0}

    det = []; mis = []
    for ev in events:
        si = ev['s']; lt = None; found = False
        for j in range(si - 1, max(-1, si - 15), -1):
            if j < 0: break
            if yp[j] in ('YELLOW', 'RED'): lt = si - j; found = True
            else:
                if found: break
        if not found:
            if yp[si] in ('YELLOW', 'RED'): lt = 0; found = True

        if found:
            det.append({'date': str(ev['date']), 'days': ev['d'], 'lt': lt})
        else:
            mis.append({'date': str(ev['date']), 'days': ev['d']})

    nd = len(det); nm = len(mis)
    dr = nd / (nd + nm) * 100 if (nd + nm) > 0 else 0

    result = {'target_class': used_class, 'n_events': len(events),
              'n_detected': nd, 'n_missed': nm, 'detection_rate': round(dr, 1),
              'events_detected': det, 'events_missed': mis}

    lts = [d['lt'] for d in det if d['lt'] is not None]
    if lts:
        result['mean_lead_time'] = round(float(np.mean(lts)), 1)
        result['min_lead_time'] = int(min(lts))
        result['max_lead_time'] = int(max(lts))

    return result


# =============================================================================
# FULL EVALUATION — prints, plots, saves JSON
# =============================================================================

def run_full_evaluation(forecast_results, df_alerts, naive_maes, output_dir, targets=None):
    """
    Run complete performance evaluation.
    Call this from any S1-S4 script after generating forecasts and alerts.

    Args:
        forecast_results: dict of {param: DataFrame}
        df_alerts: DataFrame with alert_status and alert_actual columns
        naive_maes: dict of {param: float}
        output_dir: where to save plots and JSON
        targets: optional dict of performance targets

    Returns:
        dict with all metrics
    """
    if targets is None:
        targets = {'MAPE_1day': 10.0, 'MAPE_5day': 15.0, 'MASE': 1.0, 'Lead_Time_Avg': 3.0}

    os.makedirs(output_dir, exist_ok=True)

    # ── 4.1 FORECAST ACCURACY ──
    print('\n' + '=' * 70)
    print(' SECTION 4.1: FORECAST ACCURACY')
    print(' MAE / RMSE / MAPE / MASE × 3 params × 2 horizons')
    print('=' * 70)

    accuracy_results = compute_full_metrics(forecast_results, naive_maes)

    for param in PARAMS:
        if param not in accuracy_results: continue
        r = accuracy_results[param]
        u = PARAM_UNITS[param]
        print(f'\n  {PARAM_LABELS[param]} ({u})')
        print(f'  {"─"*34}')
        print(f'  {"Metric":<8} {"1-Day":>10} {"5-Day":>10}')
        print(f'  {"MAE":<8} {r["MAE_1day"]:>9.3f} {r["MAE_5day"]:>9.3f}  {u}')
        print(f'  {"RMSE":<8} {r["RMSE_1day"]:>9.3f} {r["RMSE_5day"]:>9.3f}  {u}')
        print(f'  {"MAPE":<8} {r["MAPE_1day"]:>8.2f}% {r["MAPE_5day"]:>8.2f}%')
        print(f'  {"MASE":<8} {r["MASE_1day"]:>9.3f} {r["MASE_5day"]:>9.3f}  '
              f'{"✓" if r["MASE_1day"] < 1 else "✗"}')

    # Summary table
    if accuracy_results:
        print(f'\n  {"─"*76}')
        print(f'  {"Parameter":<22} {"MAE":>6} {"RMSE":>6} {"MAPE":>6} {"MASE":>6}   '
              f'{"MAE":>6} {"RMSE":>6} {"MAPE":>6} {"MASE":>6}')
        print(f'  {"":22s} {"── 1-Day ──":^25}   {"── 5-Day ──":^25}')
        print(f'  {"─"*76}')
        for p in PARAMS:
            if p not in accuracy_results: continue
            r = accuracy_results[p]; name = p.replace('_', ' ').title()[:21]
            print(f'  {name:<22} {r["MAE_1day"]:>5.3f} {r["RMSE_1day"]:>5.3f} '
                  f'{r["MAPE_1day"]:>4.1f}% {r["MASE_1day"]:>5.3f}   '
                  f'{r["MAE_5day"]:>5.3f} {r["RMSE_5day"]:>5.3f} '
                  f'{r["MAPE_5day"]:>4.1f}% {r["MASE_5day"]:>5.3f}')
        avg = lambda k: np.nanmean([accuracy_results[p][k] for p in accuracy_results
                                     if accuracy_results[p].get(k) is not None and not (isinstance(accuracy_results[p].get(k), float) and np.isnan(accuracy_results[p].get(k)))])
        print(f'  {"─"*76}')
        print(f'  {"Average":<22} {avg("MAE_1day"):>5.3f} {avg("RMSE_1day"):>5.3f} '
              f'{avg("MAPE_1day"):>4.1f}% {avg("MASE_1day"):>5.3f}   '
              f'{avg("MAE_5day"):>5.3f} {avg("RMSE_5day"):>5.3f} '
              f'{avg("MAPE_5day"):>4.1f}% {avg("MASE_5day"):>5.3f}')
        print(f'  {"─"*76}')

    # ── 4.2 ALERT CLASSIFICATION ──
    print('\n' + '=' * 70)
    print(' SECTION 4.2: ALERT CLASSIFICATION')
    print('=' * 70)

    classification_results = compute_alert_metrics(df_alerts)
    cm = classification_results['confusion_matrix']
    met = classification_results['class_metrics']
    acc = classification_results['accuracy']
    n_total = classification_results['n_total']

    print(f'\n  Total points: {n_total}\n')
    print(f'  {"Fc ↓ / Act →":>22}' + ''.join(f'{c:>10}' for c in ALERT_CLASSES))
    print(f'  {"─"*55}')
    for fc in ALERT_CLASSES:
        row = f'  {fc:>22}'
        for ac in ALERT_CLASSES: row += f'{cm[fc][ac]:>10}'
        print(row)
    print(f'  {"─"*55}')

    print(f'\n  {"Class":<10} {"Prec":>8} {"Recall":>8} {"F1":>8} {"Support":>8} {"Pred":>8}')
    print(f'  {"─"*55}')
    for cls in ALERT_CLASSES:
        m = met[cls]
        print(f'  {cls:<10} {m["precision"]:>7.1f}% {m["recall"]:>7.1f}% {m["f1"]:>7.1f}% '
              f'{m["support"]:>8} {m["predicted"]:>8}')
    print(f'  {"─"*55}')
    print(f'  {"Accuracy":<10} {acc:>40.1f}%')

    # ── 4.3 LEAD TIME ──
    print('\n' + '=' * 70)
    print(' SECTION 4.3: LEAD TIME')
    print('=' * 70)

    lt_results = compute_lead_time(df_alerts, 'RED', 'YELLOW')

    print(f'\n  Target class: {lt_results["target_class"]}')
    print(f'  Events: {lt_results["n_events"]}')
    for d in lt_results.get('events_detected', []):
        print(f'    ✓ {d["date"]} ({d["days"]}d) → {d["lt"]}d lead')
    for m in lt_results.get('events_missed', []):
        print(f'    ✗ {m["date"]} ({m["days"]}d) → MISSED')
    print(f'\n  Detection rate: {lt_results["detection_rate"]}%')
    if lt_results.get('mean_lead_time') is not None:
        print(f'  Lead time: mean={lt_results["mean_lead_time"]}d  '
              f'min={lt_results["min_lead_time"]}d  max={lt_results["max_lead_time"]}d')

    # ── 4.4 HYPOTHESIS VALIDATION ──
    print('\n' + '=' * 70)
    print(' SECTION 4.4: HYPOTHESIS VALIDATION')
    print('=' * 70)

    hc = hp = 0

    if accuracy_results:
        am1 = np.nanmean([accuracy_results[p]['MAPE_1day'] for p in accuracy_results
                           if accuracy_results[p].get('MAPE_1day') is not None])
        am5 = np.nanmean([accuracy_results[p]['MAPE_5day'] for p in accuracy_results
                           if accuracy_results[p].get('MAPE_5day') is not None])
        amase = np.nanmean([accuracy_results[p]['MASE_1day'] for p in accuracy_results])
        h1 = am1 < targets['MAPE_1day'] and am5 < targets['MAPE_5day']
        print(f'\n  H1 Forecast Accuracy : {"MET" if h1 else "NOT MET":>10}  '
              f'(MAPE 1d={am1:.1f}% 5d={am5:.1f}% MASE={amase:.3f})')
        hc += 1; hp += h1

    if classification_results:
        h2 = acc >= 70.0
        rr = met.get('RED', {}).get('recall', 0)
        yr = met.get('YELLOW', {}).get('recall', 0)
        print(f'  H2 Alert Classification: {"MET" if h2 else "NOT MET":>10}  '
              f'(acc={acc:.1f}% RED_r={rr:.1f}% YEL_r={yr:.1f}%)')
        hc += 1; hp += h2

    ml = lt_results.get('mean_lead_time', 0)
    h3 = ml >= targets['Lead_Time_Avg']
    print(f'  H3 Lead Time          : {"MET" if h3 else "NOT MET":>10}  '
          f'(mean={ml:.1f}d target≥{targets["Lead_Time_Avg"]}d)')
    hc += 1; hp += h3

    print(f'\n  OVERALL: {hp}/{hc} hypotheses validated')
    print('=' * 70)

    # ── PLOTS ──
    # Forecast accuracy bar chart
    if accuracy_results:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        ps = list(accuracy_results.keys())
        lb = [p.replace('_', '\n').title() for p in ps]
        for ax, m, t in zip(axes, ['MAE','RMSE','MAPE','MASE'], ['MAE','RMSE','MAPE (%)','MASE']):
            v1 = [accuracy_results[p][f'{m}_1day'] for p in ps]
            v5 = [accuracy_results[p][f'{m}_5day'] for p in ps]
            x = np.arange(len(ps)); w = 0.35
            ax.bar(x-w/2, v1, w, label='1-Day', color='#3B82F6', alpha=0.85)
            ax.bar(x+w/2, v5, w, label='5-Day', color='#EF4444', alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(lb, fontsize=8)
            ax.set_title(t, fontsize=10, fontweight='bold')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
            if m == 'MASE': ax.axhline(1.0, color='black', ls='--', lw=1.2)
        plt.suptitle('Forecast Accuracy', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir + 'forecast_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Confusion matrix heatmap
    matrix = np.array([[cm.get(fc, {}).get(ac, 0) for ac in ALERT_CLASSES] for fc in ALERT_CLASSES])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(ALERT_CLASSES); ax.set_yticklabels(ALERT_CLASSES)
    ax.set_xlabel('Actual'); ax.set_ylabel('Forecast')
    ax.set_title('Alert Classification Confusion Matrix')
    for i in range(4):
        for j in range(4):
            v = matrix[i, j]; c = 'white' if v > matrix.max() * 0.5 else 'black'
            ax.text(j, i, str(v), ha='center', va='center', color=c, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir + 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\n  ✓ {output_dir}forecast_accuracy.png')
    print(f'  ✓ {output_dir}confusion_matrix.png')

    # ── SAVE JSON ──
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, pd.Timestamp): return str(o)
        return str(o)

    all_results = {
        'forecast_accuracy': accuracy_results,
        'classification': {
            'confusion_matrix': cm, 'class_metrics': met,
            'accuracy': acc, 'n_total': n_total,
        },
        'lead_time': {k: v for k, v in lt_results.items()
                      if k not in ('events_detected', 'events_missed')},
        'targets': targets,
        'hypothesis': {'H1': bool(h1) if accuracy_results else None,
                       'H2': bool(h2) if classification_results else None,
                       'H3': bool(h3),
                       'overall': f'{hp}/{hc}'},
    }

    with open(output_dir + 'performance_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=conv)
    print(f'  ✓ {output_dir}performance_metrics.json')

    return all_results
