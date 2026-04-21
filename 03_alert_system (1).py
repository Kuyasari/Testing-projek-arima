"""
03_alert_system.py

VRU COMPRESSOR - RULE-BASED ALERT CLASSIFICATION SYSTEM
Implements the thesis alert logic: OFF / GREEN / YELLOW / RED

Classification Rules (from thesis proposal):
    OFF    : Discharge temp < 90°F (compressor not running)
    GREEN  : All 3 parameters within normal range (0 violations)
    YELLOW : Exactly 1 parameter out of normal range (1 violation)
    RED    : 2+ parameters out of range OR critical violation (temp > 300°F)

Thresholds (Ro-Flo manufacturer specs + operational experience):
    Discharge Temperature:
        Normal:   110 - 150 °F
        Warning:  > 150 °F  (performance degradation)
        Critical: > 300 °F  (immediate action — thermal runaway)
        OFF:      < 90 °F   (compressor not running)
    
    Discharge Pressure:
        Normal:   10 - 30 psi
        Low:      < 10 psi  (vane wear / internal leakage)
        High:     > 30 psi  (downstream restriction)
    
    Jacket Water Pressure:
        Normal:   12 - 20 psi
        Low:      < 12 psi  (cooling system degradation)
        High:     > 20 psi  (flow restriction)

Author: Kaa Albaraq Sakha
Thesis: Time-Series Forecasting for Preventive Maintenance of VRU Compressors
Institution: Universitas Gadjah Mada
Date: April 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# =============================================================================
# CONFIGURATION — THESIS THRESHOLDS
# =============================================================================

# Validated through:
# - Ro-Flo manufacturer specifications (Ro-Flo Compressors, n.d.)
# - 52-month facility operational data (Aug 2021 – Dec 2025)
# - Mentor operational experience
# - Empirical analysis of 28 recorded failures

THRESHOLDS = {
    'discharge_temp': {
        'OFF':      90,          # °F — below this, compressor is not running
        'NORMAL':   (110, 150),  # °F — good condition
        'WARNING':  150,         # °F — performance degradation begins
        'CRITICAL': 300,         # °F — thermal runaway, immediate action
    },
    'discharge_pressure': {
        'NORMAL':   (10, 30),    # psi — adequate compression
        'LOW':      10,          # psi — below: vane wear / internal leakage
        'HIGH':     30,          # psi — above: downstream restriction
    },
    'jacket_water': {
        'NORMAL':   (12, 20),    # psi — adequate cooling
        'LOW':      12,          # psi — below: cooling system degradation
        'HIGH':     20,          # psi — above: flow restriction
    }
}

INPUT_DIR = 'arima_models/'
OUTPUT_DIR = 'alerts/'

# =============================================================================
# STEP 1: CHECK INDIVIDUAL PARAMETER VIOLATIONS
# =============================================================================

def check_discharge_temp(value):
    """
    Check discharge temperature against thresholds.
    
    Returns:
        (is_violation: bool, is_critical: bool, description: str)
    
    Rules:
        < 90°F   → OFF (not a violation, compressor not running)
        110-150°F → Normal (no violation)
        > 150°F   → Warning violation
        > 300°F   → CRITICAL violation (auto-RED regardless of other params)
    """
    if value < THRESHOLDS['discharge_temp']['OFF']:
        # Not a violation — compressor is OFF
        return False, False, 'OFF'
    
    if value > THRESHOLDS['discharge_temp']['CRITICAL']:
        return True, True, f'CRITICAL: Temp {value:.1f}°F (>{THRESHOLDS["discharge_temp"]["CRITICAL"]}°F — thermal runaway)'
    
    if value > THRESHOLDS['discharge_temp']['WARNING']:
        return True, False, f'Temp HIGH {value:.1f}°F (>{THRESHOLDS["discharge_temp"]["WARNING"]}°F — degradation zone)'
    
    lo, hi = THRESHOLDS['discharge_temp']['NORMAL']
    if lo <= value <= hi:
        return False, False, f'Temp OK {value:.1f}°F (normal {lo}-{hi}°F)'
    
    # Below normal but above OFF (90-110°F range — startup/warmup)
    return False, False, f'Temp {value:.1f}°F (startup/warmup zone)'


def check_discharge_pressure(value):
    """
    Check discharge pressure against thresholds.
    
    Returns:
        (is_violation: bool, is_critical: bool, description: str)
    
    Rules:
        10-30 psi → Normal
        < 10 psi  → Violation (insufficient compression — vane wear)
        > 30 psi  → Violation (downstream restriction)
    """
    lo, hi = THRESHOLDS['discharge_pressure']['NORMAL']
    
    if value < THRESHOLDS['discharge_pressure']['LOW']:
        return True, False, f'Press LOW {value:.1f} psi (<{THRESHOLDS["discharge_pressure"]["LOW"]} psi — vane wear / leakage)'
    
    if value > THRESHOLDS['discharge_pressure']['HIGH']:
        return True, False, f'Press HIGH {value:.1f} psi (>{THRESHOLDS["discharge_pressure"]["HIGH"]} psi — downstream restriction)'
    
    return False, False, f'Press OK {value:.1f} psi (normal {lo}-{hi} psi)'


def check_jacket_water(value):
    """
    Check jacket water pressure against thresholds.
    
    Returns:
        (is_violation: bool, is_critical: bool, description: str)
    
    Rules:
        12-20 psi → Normal
        < 12 psi  → Violation (cooling system degradation — LEADING INDICATOR)
        > 20 psi  → Violation (flow restriction)
    
    Note: Jacket water is a leading indicator.
    Due to 7-14 day thermal lag, jacket water pressure declines
    BEFORE discharge temperature rises. A jacket water violation
    today predicts a temperature violation 7-14 days later.
    """
    lo, hi = THRESHOLDS['jacket_water']['NORMAL']
    
    if value < THRESHOLDS['jacket_water']['LOW']:
        return True, False, f'Jacket LOW {value:.1f} psi (<{THRESHOLDS["jacket_water"]["LOW"]} psi — cooling degradation, LEADING INDICATOR)'
    
    if value > THRESHOLDS['jacket_water']['HIGH']:
        return True, False, f'Jacket HIGH {value:.1f} psi (>{THRESHOLDS["jacket_water"]["HIGH"]} psi — flow restriction)'
    
    return False, False, f'Jacket OK {value:.1f} psi (normal {lo}-{hi} psi)'


# =============================================================================
# STEP 2: CLASSIFY ALERT — THE CORE THESIS RULE
# =============================================================================

def classify_alert(temp_value, pressure_value, jacket_value):
    """
    Apply the thesis alert classification rules.
    
    This is the complete rule set from the thesis proposal:
    
        STEP 1: Check if compressor is running
            IF discharge_temp < 90°F → OFF (skip further checks)
        
        STEP 2: Check each parameter for violations
            Discharge Temp:    violation if > 150°F, CRITICAL if > 300°F
            Discharge Pressure: violation if < 10 psi OR > 30 psi
            Jacket Water:      violation if < 12 psi OR > 20 psi
        
        STEP 3: Count violations and classify
            Any CRITICAL violation   → RED  (regardless of count)
            2+ violations            → RED
            1 violation              → YELLOW
            0 violations             → GREEN
    
    Returns:
        dict with keys:
            'status':       'OFF' | 'GREEN' | 'YELLOW' | 'RED'
            'violations':   list of violation descriptions
            'n_violations': int
            'is_critical':  bool
            'details':      dict of individual parameter results
            'reason':       human-readable explanation
    """
    
    # --- Step 1: Check if compressor is running ---
    if temp_value < THRESHOLDS['discharge_temp']['OFF']:
        return {
            'status': 'OFF',
            'violations': [],
            'n_violations': 0,
            'is_critical': False,
            'details': {
                'discharge_temp': 'OFF',
                'discharge_pressure': 'N/A',
                'jacket_water': 'N/A'
            },
            'reason': f'Compressor not running (temp {temp_value:.1f}°F < {THRESHOLDS["discharge_temp"]["OFF"]}°F)'
        }
    
    # --- Step 2: Check each parameter ---
    temp_violation, temp_critical, temp_desc = check_discharge_temp(temp_value)
    press_violation, press_critical, press_desc = check_discharge_pressure(pressure_value)
    jacket_violation, jacket_critical, jacket_desc = check_jacket_water(jacket_value)
    
    violations = []
    is_critical = False
    
    if temp_violation:
        violations.append(temp_desc)
        if temp_critical:
            is_critical = True
    
    if press_violation:
        violations.append(press_desc)
        if press_critical:
            is_critical = True
    
    if jacket_violation:
        violations.append(jacket_desc)
        if jacket_critical:
            is_critical = True
    
    n_violations = len(violations)
    
    # --- Step 3: Classify ---
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
    
    # Individual parameter status for logging
    details = {
        'discharge_temp': 'CRITICAL' if temp_critical else ('VIOLATION' if temp_violation else 'OK'),
        'discharge_pressure': 'VIOLATION' if press_violation else 'OK',
        'jacket_water': 'VIOLATION' if jacket_violation else 'OK'
    }
    
    return {
        'status': status,
        'violations': violations,
        'n_violations': n_violations,
        'is_critical': is_critical,
        'details': details,
        'reason': reason
    }


# =============================================================================
# STEP 3: GENERATE ALERTS FOR FORECAST DATA
# =============================================================================

def generate_alerts():
    """
    Apply alert classification to all ARIMA forecast points.
    
    Uses 5-day-ahead forecasts for classification because:
    - 5-day horizon provides actionable lead time for maintenance planning
    - Aligns with the thesis research question (preventive maintenance)
    - Matches the intervention planning window at the facility
    
    Also classifies 1-day forecasts for comparison.
    """
    print("="*70)
    print(" VRU COMPRESSOR ALERT CLASSIFICATION SYSTEM")
    print(" Rules: OFF / GREEN / YELLOW / RED")
    print("="*70)
    print()
    
    # --- Load forecasts ---
    print("  Loading ARIMA forecast data...")
    
    forecasts = {}
    for param in ['discharge_temp', 'discharge_pressure', 'jacket_water']:
        filepath = f'{INPUT_DIR}forecasts_{param}.csv'
        if not os.path.exists(filepath):
            print(f"  ERROR: {filepath} not found — run 02_arima_modeling.py first")
            return None
        forecasts[param] = pd.read_csv(filepath)
    
    # Check alignment
    n_rows = min(len(forecasts[p]) for p in forecasts)
    print(f"  Loaded {n_rows} forecast points per parameter")
    print()
    
    # --- Display thresholds ---
    print("  Thresholds Applied:")
    print(f"    Discharge Temp:  Normal {THRESHOLDS['discharge_temp']['NORMAL']}°F "
          f"| Warning >{THRESHOLDS['discharge_temp']['WARNING']}°F "
          f"| Critical >{THRESHOLDS['discharge_temp']['CRITICAL']}°F "
          f"| OFF <{THRESHOLDS['discharge_temp']['OFF']}°F")
    print(f"    Discharge Press: Normal {THRESHOLDS['discharge_pressure']['NORMAL']} psi "
          f"| Low <{THRESHOLDS['discharge_pressure']['LOW']} psi "
          f"| High >{THRESHOLDS['discharge_pressure']['HIGH']} psi")
    print(f"    Jacket Water:    Normal {THRESHOLDS['jacket_water']['NORMAL']} psi "
          f"| Low <{THRESHOLDS['jacket_water']['LOW']} psi "
          f"| High >{THRESHOLDS['jacket_water']['HIGH']} psi")
    print()
    
    print("  Classification Rules:")
    print("    OFF    : Temp < 90°F (compressor not running)")
    print("    GREEN  : 0 violations (all parameters normal)")
    print("    YELLOW : 1 violation (one parameter out of range)")
    print("    RED    : 2+ violations OR any critical violation (temp > 300°F)")
    print()
    
    # --- Classify each forecast point ---
    alert_results = []
    
    for i in range(n_rows):
        # 5-day forecast values
        temp_5d = forecasts['discharge_temp']['forecast_5day'].iloc[i]
        press_5d = forecasts['discharge_pressure']['forecast_5day'].iloc[i]
        jacket_5d = forecasts['jacket_water']['forecast_5day'].iloc[i]
        
        # 1-day forecast values
        temp_1d = forecasts['discharge_temp']['forecast_1day'].iloc[i]
        press_1d = forecasts['discharge_pressure']['forecast_1day'].iloc[i]
        jacket_1d = forecasts['jacket_water']['forecast_1day'].iloc[i]
        
        # Actual values (for retrospective comparison)
        temp_actual = forecasts['discharge_temp']['actual_5day'].iloc[i]
        press_actual = forecasts['discharge_pressure']['actual_5day'].iloc[i]
        jacket_actual = forecasts['jacket_water']['actual_5day'].iloc[i]
        
        # Classify 5-day forecast (primary — used for alert generation)
        result_5d = classify_alert(temp_5d, press_5d, jacket_5d)
        
        # Classify 1-day forecast (secondary — for comparison)
        result_1d = classify_alert(temp_1d, press_1d, jacket_1d)
        
        # Classify actual values (ground truth — for detection rate)
        result_actual = classify_alert(temp_actual, press_actual, jacket_actual)
        
        alert_results.append({
            # Dates
            'forecast_date': forecasts['discharge_temp']['forecast_date'].iloc[i],
            'target_date': forecasts['discharge_temp']['target_date'].iloc[i],
            
            # 5-day forecast alert (PRIMARY)
            'alert_status': result_5d['status'],
            'alert_reason': result_5d['reason'],
            'n_violations': result_5d['n_violations'],
            'is_critical': result_5d['is_critical'],
            
            # Individual parameter status (5-day)
            'temp_status': result_5d['details']['discharge_temp'],
            'pressure_status': result_5d['details']['discharge_pressure'],
            'jacket_status': result_5d['details']['jacket_water'],
            
            # 1-day forecast alert (for comparison)
            'alert_1day': result_1d['status'],
            
            # Actual alert (ground truth for validation)
            'alert_actual': result_actual['status'],
            
            # Forecast values
            'temp_forecast_5d': round(temp_5d, 2),
            'press_forecast_5d': round(press_5d, 2),
            'jacket_forecast_5d': round(jacket_5d, 2),
            
            # Actual values
            'temp_actual': round(temp_actual, 2),
            'press_actual': round(press_actual, 2),
            'jacket_actual': round(jacket_actual, 2),
            
            # Violations list (for debugging)
            'violations': '; '.join(result_5d['violations']) if result_5d['violations'] else ''
        })
    
    alerts_df = pd.DataFrame(alert_results)
    
    # --- Display results ---
    print("  Alert Distribution (5-day forecast):")
    print("  " + "-"*50)
    total = len(alerts_df)
    for status in ['GREEN', 'YELLOW', 'RED', 'OFF']:
        count = (alerts_df['alert_status'] == status).sum()
        pct = count / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"    {status:6s}: {count:4d} ({pct:5.1f}%)  {bar}")
    print("  " + "-"*50)
    print()
    
    # Alert distribution for actual values (ground truth)
    print("  Alert Distribution (actual values — ground truth):")
    print("  " + "-"*50)
    for status in ['GREEN', 'YELLOW', 'RED', 'OFF']:
        count = (alerts_df['alert_actual'] == status).sum()
        pct = count / total * 100 if total > 0 else 0
        print(f"    {status:6s}: {count:4d} ({pct:5.1f}%)")
    print("  " + "-"*50)
    print()
    
    # Forecast vs actual agreement
    agreement = (alerts_df['alert_status'] == alerts_df['alert_actual']).sum()
    agreement_pct = agreement / total * 100 if total > 0 else 0
    print(f"  Forecast-Actual Agreement: {agreement}/{total} ({agreement_pct:.1f}%)")
    print()
    
    # Show non-GREEN alerts
    non_green = alerts_df[~alerts_df['alert_status'].isin(['GREEN'])]
    if len(non_green) > 0:
        print(f"  Non-GREEN Alerts ({len(non_green)} total):")
        for _, row in non_green.iterrows():
            print(f"    {row['forecast_date']} → {row['alert_status']:6s} | {row['alert_reason']}")
            if row['violations']:
                for v in row['violations'].split('; '):
                    print(f"      • {v}")
    else:
        print("  All alerts are GREEN (no violations detected in test period)")
    print()
    
    # --- Save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    alerts_df.to_csv(f'{OUTPUT_DIR}alerts_generated.csv', index=False)
    print(f"  ✓ Alerts saved: {OUTPUT_DIR}alerts_generated.csv")
    
    # Top reasons
    print()
    print("  Top Alert Reasons:")
    for reason, count in alerts_df['alert_reason'].value_counts().head(5).items():
        print(f"    [{count:3d}] {reason}")
    
    print()
    
    return alerts_df


# =============================================================================
# STEP 4: VISUALIZATION
# =============================================================================

def plot_alert_timeline(alerts_df):
    """
    Create alert timeline visualization with 4 states.
    """
    alerts_df = alerts_df.copy()
    alerts_df['forecast_dt'] = pd.to_datetime(alerts_df['forecast_date'])
    
    status_map = {'OFF': -1, 'GREEN': 0, 'YELLOW': 1, 'RED': 2}
    colors_map = {'OFF': '#9CA3AF', 'GREEN': '#16A34A', 'YELLOW': '#EAB308', 'RED': '#DC2626'}
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    
    # --- Panel 1: Alert Timeline ---
    ax = axes[0]
    alerts_df['status_num'] = alerts_df['alert_status'].map(status_map)
    
    for status, color in colors_map.items():
        mask = alerts_df['alert_status'] == status
        if mask.any():
            ax.scatter(alerts_df.loc[mask, 'forecast_dt'],
                      alerts_df.loc[mask, 'status_num'],
                      c=color, s=60, alpha=0.8, label=status, edgecolors='white', linewidth=0.5)
    
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_yticklabels(['OFF\n(Not running)', 'GREEN\n(Normal)', 'YELLOW\n(1 violation)', 'RED\n(2+ or critical)'])
    ax.set_title('VRU Compressor Alert Classification Timeline', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(alerts_df['forecast_dt'].min() - pd.Timedelta(days=1),
                alerts_df['forecast_dt'].max() + pd.Timedelta(days=1))
    
    # --- Panel 2: Discharge Temperature ---
    ax = axes[1]
    ax.plot(alerts_df['forecast_dt'], alerts_df['temp_forecast_5d'], 'r-', linewidth=1, label='Forecast (5d)', alpha=0.8)
    ax.plot(alerts_df['forecast_dt'], alerts_df['temp_actual'], 'b-', linewidth=1, label='Actual', alpha=0.6)
    ax.axhline(THRESHOLDS['discharge_temp']['WARNING'], color='orange', linestyle='--', linewidth=0.8, label=f'Warning ({THRESHOLDS["discharge_temp"]["WARNING"]}°F)')
    ax.axhspan(THRESHOLDS['discharge_temp']['NORMAL'][0], THRESHOLDS['discharge_temp']['NORMAL'][1],
               alpha=0.1, color='green', label=f'Normal ({THRESHOLDS["discharge_temp"]["NORMAL"][0]}-{THRESHOLDS["discharge_temp"]["NORMAL"][1]}°F)')
    ax.set_ylabel('Temp (°F)')
    ax.set_title('Discharge Temperature', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Panel 3: Discharge Pressure ---
    ax = axes[2]
    ax.plot(alerts_df['forecast_dt'], alerts_df['press_forecast_5d'], 'r-', linewidth=1, label='Forecast (5d)', alpha=0.8)
    ax.plot(alerts_df['forecast_dt'], alerts_df['press_actual'], 'b-', linewidth=1, label='Actual', alpha=0.6)
    ax.axhspan(THRESHOLDS['discharge_pressure']['NORMAL'][0], THRESHOLDS['discharge_pressure']['NORMAL'][1],
               alpha=0.1, color='green', label=f'Normal ({THRESHOLDS["discharge_pressure"]["NORMAL"][0]}-{THRESHOLDS["discharge_pressure"]["NORMAL"][1]} psi)')
    ax.set_ylabel('Press (psi)')
    ax.set_title('Discharge Pressure', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Panel 4: Jacket Water ---
    ax = axes[3]
    ax.plot(alerts_df['forecast_dt'], alerts_df['jacket_forecast_5d'], 'r-', linewidth=1, label='Forecast (5d)', alpha=0.8)
    ax.plot(alerts_df['forecast_dt'], alerts_df['jacket_actual'], 'b-', linewidth=1, label='Actual', alpha=0.6)
    ax.axhspan(THRESHOLDS['jacket_water']['NORMAL'][0], THRESHOLDS['jacket_water']['NORMAL'][1],
               alpha=0.1, color='green', label=f'Normal ({THRESHOLDS["jacket_water"]["NORMAL"][0]}-{THRESHOLDS["jacket_water"]["NORMAL"][1]} psi)')
    ax.axhline(THRESHOLDS['jacket_water']['LOW'], color='red', linestyle='--', linewidth=0.8, label=f'Low ({THRESHOLDS["jacket_water"]["LOW"]} psi)')
    ax.set_ylabel('Press (psi)')
    ax.set_title('Jacket Water Pressure (Leading Indicator)', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}alert_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Alert timeline plot saved: {OUTPUT_DIR}alert_timeline.png")


def plot_confusion_matrix(alerts_df):
    """
    Compare forecast alerts vs actual alerts (confusion-style breakdown).
    Shows how well the forecast-based classification matches reality.
    """
    statuses = ['OFF', 'GREEN', 'YELLOW', 'RED']
    
    print()
    print("  Forecast vs Actual Alert Confusion Matrix:")
    print("  " + "-"*55)
    header = f"  {'':14s}" + "".join(f"{'Act:'+s:>10s}" for s in statuses)
    print(header)
    print("  " + "-"*55)
    
    for fc_status in statuses:
        row = f"  {'Fc:'+fc_status:14s}"
        for act_status in statuses:
            count = ((alerts_df['alert_status'] == fc_status) & 
                     (alerts_df['alert_actual'] == act_status)).sum()
            row += f"{count:>10d}"
        print(row)
    print("  " + "-"*55)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Execute the complete alert classification system.
    """
    print()
    
    # Generate alerts
    alerts_df = generate_alerts()
    
    if alerts_df is None:
        print("  Exiting — no forecast data available")
        return
    
    # Visualize
    plot_alert_timeline(alerts_df)
    plot_confusion_matrix(alerts_df)
    
    # Save threshold configuration
    with open(f'{OUTPUT_DIR}threshold_config.json', 'w') as f:
        # Convert tuples to lists for JSON
        config = {}
        for param, thresholds in THRESHOLDS.items():
            config[param] = {}
            for key, val in thresholds.items():
                config[param][key] = list(val) if isinstance(val, tuple) else val
        json.dump(config, f, indent=2)
    
    print(f"  ✓ Threshold config saved: {OUTPUT_DIR}threshold_config.json")
    print()
    print("="*70)
    print(" ALERT CLASSIFICATION COMPLETE")
    print("="*70)
    print()
    print("  Next step: Run 04_performance_evaluation.py to calculate metrics")
    print()


if __name__ == "__main__":
    main()
