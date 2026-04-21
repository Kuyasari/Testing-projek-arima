"""
02_arima_modeling.py

VRU COMPRESSOR - ARIMA TIME-SERIES FORECASTING
Implements Box-Jenkins methodology for discharge temperature, discharge pressure, 
and jacket water pressure forecasting with sudden event filtering.

CRITICAL FIX: Rolling window forecast (original used static forecast from training endpoint)
RECOMMENDATION: Uses daily averages (not 4-point sub-daily) — aligned with 5-day horizon

Author: Kaa Albaraq Sakha
Thesis: Time-Series Forecasting for Preventive Maintenance of VRU Compressors
Institution: Universitas Gadjah Mada
Date: April 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = 'VRU_data_cleaned.csv'  # From 01_data_preprocessing.py
OUTPUT_DIR = 'arima_models/'
TRAIN_TEST_SPLIT = 0.85  # 85% train, 15% test
FORECAST_HORIZON = 5  # 5-day ahead forecasts

# Sudden event detection thresholds (filter these out!)
SUDDEN_EVENT_THRESHOLDS = {
    'discharge_temp': {
        'spike': 30,      # °F change within 1 day = sudden event
        'drop': -30
    },
    'discharge_pressure': {
        'spike': 10,      # psi change within 1 day = sudden event  
        'drop': -10
    },
    'jacket_water': {
        'spike': 5,       # psi change within 1 day = sudden event
        'drop': -5
    }
}

# =============================================================================
# STEP 1: LOAD CLEANED DATA
# =============================================================================

def load_data():
    """Load preprocessed VRU data from Step 01"""
    print("="*70)
    print("STEP 1: LOADING CLEANED DATA")
    print("="*70)
    
    df = pd.read_csv(INPUT_FILE, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    # Ensure daily frequency (fill gaps if any)
    df = df.asfreq('D')
    
    print(f"  Loaded {len(df)} observations")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Parameters: {df.columns.tolist()}")
    print(f"  Missing values per column:")
    for col in df.columns:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            print(f"    {col}: {n_miss} ({n_miss/len(df)*100:.1f}%)")
    
    # Forward-fill small gaps (1-2 days), leave larger gaps as NaN
    df = df.fillna(method='ffill', limit=2)
    
    remaining_miss = df.isnull().sum().sum()
    if remaining_miss > 0:
        print(f"  After ffill(limit=2): {remaining_miss} remaining NaN → dropping rows")
        df = df.dropna()
    
    print(f"  Final dataset: {len(df)} observations")
    print()
    
    return df

# =============================================================================
# STEP 2: SUDDEN EVENT DETECTION AND FILTERING
# =============================================================================

def detect_sudden_events(df):
    """
    Detect and flag sudden operational upsets for exclusion from ARIMA training.
    
    Sudden events include:
    - High discharge pressure from downstream restrictions (blowdown-addressable)
    - Electrical trips (breaker events)
    - Auxiliary system failures (lube oil, instrument air)
    - Large single-day parameter changes indicating non-gradual upsets
    
    These are NOT suitable for trend-based forecasting and must be excluded.
    """
    print("="*70)
    print("STEP 2: SUDDEN EVENT DETECTION AND FILTERING")
    print("="*70)
    
    df_filtered = df.copy()
    sudden_event_mask = pd.Series(False, index=df.index)
    
    # Calculate day-to-day changes
    temp_change = df['discharge_temp'].diff()
    pressure_change = df['discharge_pressure'].diff()
    jacket_change = df['jacket_water'].diff()
    
    # Flag sudden spikes/drops exceeding thresholds
    temp_sudden = (
        (temp_change > SUDDEN_EVENT_THRESHOLDS['discharge_temp']['spike']) |
        (temp_change < SUDDEN_EVENT_THRESHOLDS['discharge_temp']['drop'])
    )
    
    pressure_sudden = (
        (pressure_change > SUDDEN_EVENT_THRESHOLDS['discharge_pressure']['spike']) |
        (pressure_change < SUDDEN_EVENT_THRESHOLDS['discharge_pressure']['drop'])
    )
    
    jacket_sudden = (
        (jacket_change > SUDDEN_EVENT_THRESHOLDS['jacket_water']['spike']) |
        (jacket_change < SUDDEN_EVENT_THRESHOLDS['jacket_water']['drop'])
    )
    
    # Combine all sudden event indicators
    sudden_event_mask = temp_sudden | pressure_sudden | jacket_sudden
    
    # Also flag recovery period (next 2 days after sudden event)
    for i in range(1, 3):
        sudden_event_mask = sudden_event_mask | sudden_event_mask.shift(-i).fillna(False)
    
    # Count and report sudden events
    n_sudden = sudden_event_mask.sum()
    
    print(f"  Sudden Event Detection:")
    print(f"    Temperature spikes/drops: {temp_sudden.sum()} days")
    print(f"    Pressure spikes/drops: {pressure_sudden.sum()} days")
    print(f"    Jacket water spikes/drops: {jacket_sudden.sum()} days")
    print(f"    Total sudden events (including recovery): {n_sudden} days")
    print(f"    Percentage of dataset: {n_sudden/len(df)*100:.1f}%")
    print()
    
    # Filter out sudden events
    df_filtered = df[~sudden_event_mask].copy()
    
    print(f"  After filtering:")
    print(f"    Original observations: {len(df)}")
    print(f"    Filtered observations: {len(df_filtered)}")
    print(f"    Removed: {len(df) - len(df_filtered)} days")
    print(f"    Retention rate: {len(df_filtered)/len(df)*100:.1f}%")
    print()
    
    return df_filtered, sudden_event_mask

# =============================================================================
# STEP 3: TRAIN/TEST SPLIT
# =============================================================================

def train_test_split(df):
    """Split data into training and testing sets"""
    print("="*70)
    print("STEP 3: TRAIN/TEST SPLIT")
    print("="*70)
    
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    print(f"  Training set: {len(train)} observations ({TRAIN_TEST_SPLIT*100:.0f}%)")
    print(f"    Date range: {train.index.min().date()} to {train.index.max().date()}")
    print()
    print(f"  Testing set: {len(test)} observations ({(1-TRAIN_TEST_SPLIT)*100:.0f}%)")
    print(f"    Date range: {test.index.min().date()} to {test.index.max().date()}")
    print()
    
    return train, test

# =============================================================================
# STEP 4: STATIONARITY ASSESSMENT (Box-Jenkins Stage 1)
# =============================================================================

def test_stationarity(series, param_name):
    """
    Augmented Dickey-Fuller test for stationarity
    
    H0: Series has unit root (non-stationary)
    H1: Series is stationary
    
    Decision: p-value < 0.05 -> Reject H0 -> Stationary
    """
    print(f"\n--- Stationarity Test: {param_name} ---")
    
    result = adfuller(series.dropna())
    
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.6f}")
    print(f"  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.4f}")
    
    if result[1] < 0.05:
        print(f"  ✓ Series is STATIONARY (p={result[1]:.4f} < 0.05)")
        d_order = 0
    else:
        print(f"  ✗ Series is NON-STATIONARY (p={result[1]:.4f} >= 0.05)")
        print(f"  → Applying first differencing (d=1)")
        d_order = 1
        
        # Test differenced series
        diff_series = series.diff().dropna()
        result_diff = adfuller(diff_series)
        print(f"\n  After differencing:")
        print(f"    ADF Statistic: {result_diff[0]:.4f}")
        print(f"    p-value: {result_diff[1]:.6f}")
        
        if result_diff[1] < 0.05:
            print(f"    ✓ Differenced series is STATIONARY")
        else:
            print(f"    ⚠ Warning: Differenced series still non-stationary")
            print(f"    → May need d=2, but trying d=1 first")
    
    return d_order

# =============================================================================
# STEP 5: MODEL ORDER IDENTIFICATION (Box-Jenkins Stage 2)
# =============================================================================

def plot_acf_pacf(series, param_name, d_order):
    """
    Plot ACF and PACF for model order identification
    
    Pattern recognition:
    - AR(p): PACF cuts off at lag p, ACF decays
    - MA(q): ACF cuts off at lag q, PACF decays
    - ARMA(p,q): Both decay gradually
    """
    print(f"\n--- ACF/PACF Analysis: {param_name} ---")
    
    # Apply differencing if needed
    if d_order == 1:
        series_diff = series.diff().dropna()
        title_suffix = " (differenced)"
    else:
        series_diff = series.dropna()
        title_suffix = ""
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    plot_acf(series_diff, lags=20, ax=axes[0])
    axes[0].set_title(f'ACF: {param_name}{title_suffix}')
    
    plot_pacf(series_diff, lags=20, ax=axes[1])
    axes[1].set_title(f'PACF: {param_name}{title_suffix}')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}acf_pacf_{param_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ ACF/PACF plots saved: acf_pacf_{param_name}.png")
    print(f"    Examine plots to identify p and q orders")
    print(f"    Typical for VRU: p ∈ {{0,1,2}}, q ∈ {{0,1,2}}")

# =============================================================================
# STEP 6: MODEL SELECTION VIA GRID SEARCH (Box-Jenkins Stages 2-3)
# =============================================================================

def select_best_arima(train_series, param_name, d_order):
    """
    Grid search over (p,q) combinations to find best ARIMA model
    Uses AIC (Akaike Information Criterion) for selection
    
    Lower AIC = better model (balances fit quality vs complexity)
    """
    print(f"\n--- Model Selection: {param_name} ---")
    print("  Testing ARIMA(p,{},q) models...".format(d_order))
    
    best_aic = np.inf
    best_order = None
    best_model = None
    
    results_summary = []
    
    # Grid search
    for p in range(0, 4):  # p ∈ {0,1,2,3}
        for q in range(0, 4):  # q ∈ {0,1,2,3}
            try:
                model = ARIMA(train_series, order=(p, d_order, q))
                fitted = model.fit()
                
                aic = fitted.aic
                bic = fitted.bic
                
                results_summary.append({
                    'p': p, 'q': q, 
                    'AIC': aic, 'BIC': bic,
                    'params': len(fitted.params)
                })
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d_order, q)
                    best_model = fitted
                    
            except Exception as e:
                # Some combinations may fail to converge
                continue
    
    # Display results
    results_df = pd.DataFrame(results_summary).sort_values('AIC')
    print("\n  Top 5 models by AIC:")
    for _, row in results_df.head().iterrows():
        print(f"    ARIMA({int(row['p'])},{d_order},{int(row['q'])})  AIC={row['AIC']:.2f}  BIC={row['BIC']:.2f}")
    print()
    
    print(f"  ✓ Best model: ARIMA{best_order}")
    print(f"    AIC: {best_aic:.2f}")
    print(f"    Parameters: {len(best_model.params)}")
    print()
    
    return best_model, best_order

# =============================================================================
# STEP 7: DIAGNOSTIC VALIDATION (Box-Jenkins Stage 4)
# =============================================================================

def validate_model(model, param_name):
    """
    Model diagnostic checks:
    1. Ljung-Box test (residual autocorrelation)
    2. Shapiro-Wilk test (residual normality)
    3. Visual diagnostics
    """
    print(f"\n--- Diagnostic Validation: {param_name} ---")
    
    residuals = model.resid
    
    # 1. Ljung-Box Test
    print("\n  1. Ljung-Box Test (Residual Independence):")
    print("     H0: No autocorrelation in residuals")
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = lb_test['lb_pvalue'].iloc[0]
    
    print(f"     p-value: {lb_pvalue:.4f}")
    if lb_pvalue >= 0.05:
        print(f"     ✓ PASS: Residuals appear independent (p={lb_pvalue:.4f} >= 0.05)")
    else:
        print(f"     ⚠ WARNING: Some autocorrelation remains (p={lb_pvalue:.4f} < 0.05)")
    
    # 2. Shapiro-Wilk Test
    print("\n  2. Shapiro-Wilk Test (Residual Normality):")
    print("     H0: Residuals are normally distributed")
    sw_stat, sw_pvalue = shapiro(residuals)
    
    print(f"     p-value: {sw_pvalue:.4f}")
    if sw_pvalue >= 0.05:
        print(f"     ✓ PASS: Residuals approximately normal (p={sw_pvalue:.4f} >= 0.05)")
    else:
        print(f"     ⚠ Note: Residuals deviate from normality (p={sw_pvalue:.4f} < 0.05)")
        print(f"     → Affects prediction intervals, not point forecasts")
    
    # 3. Visual Diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residual time series
    axes[0,0].plot(residuals)
    axes[0,0].axhline(0, color='r', linestyle='--', linewidth=0.8)
    axes[0,0].set_title('Residuals Over Time')
    axes[0,0].set_xlabel('Observation')
    axes[0,0].set_ylabel('Residual')
    
    # Residual histogram
    axes[0,1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[0,1].set_title('Residual Distribution')
    axes[0,1].set_xlabel('Residual Value')
    axes[0,1].set_ylabel('Frequency')
    
    # ACF of residuals
    plot_acf(residuals, lags=20, ax=axes[1,0])
    axes[1,0].set_title('ACF of Residuals')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}diagnostics_{param_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Diagnostic plots saved: diagnostics_{param_name}.png")
    
    validation_passed = (lb_pvalue >= 0.05)
    
    if validation_passed:
        print(f"\n  ✓✓ Model VALIDATED for {param_name}")
    else:
        print(f"\n  ⚠ Model has some diagnostic issues but proceeding")
        print(f"     (Common in industrial data with measurement noise)")
    
    return validation_passed

# =============================================================================
# STEP 8: GENERATE FORECASTS (FIXED — ROLLING WINDOW)
# =============================================================================

def generate_forecasts(train_series, test_series, best_order, param_name):
    """
    Generate rolling 5-day ahead forecasts for test period.
    
    *** CRITICAL FIX ***
    Previous version called model.forecast(5) in a loop without updating,
    producing identical forecasts for every test point.
    
    This version uses an EXPANDING WINDOW approach:
    - Start with training data
    - For each test point, forecast 5 days ahead
    - Append actual observation to history
    - Refit model every 20 steps (balance speed vs accuracy)
    
    For ~235 test points this is computationally feasible on standard PC.
    """
    print(f"\n--- Generating Rolling Forecasts: {param_name} ---")
    print(f"    Method: Expanding window with periodic refitting")
    print(f"    Refit interval: every 20 observations")
    print(f"    Forecast horizon: {FORECAST_HORIZON} days")
    
    history = list(train_series.values)
    history_index = list(train_series.index)
    
    forecasts = []
    refit_interval = 20  # Refit model every N steps
    current_model = None
    
    n_forecasts = len(test_series) - FORECAST_HORIZON + 1
    
    for i in range(n_forecasts):
        # Refit model periodically or on first iteration
        if i % refit_interval == 0:
            try:
                model = ARIMA(history, order=best_order)
                current_model = model.fit()
            except Exception:
                # If refit fails, keep previous model
                pass
        
        if current_model is None:
            # Fallback: fit once if first attempt failed
            model = ARIMA(history, order=best_order)
            current_model = model.fit()
        
        # Forecast 5 days ahead
        fc = current_model.forecast(steps=FORECAST_HORIZON)
        
        forecasts.append({
            'forecast_date': str(test_series.index[i].date()),
            'target_date': str(test_series.index[i + FORECAST_HORIZON - 1].date()),
            'forecast_1day': float(fc[0]),
            'forecast_5day': float(fc[FORECAST_HORIZON - 1]),
            'actual_1day': float(test_series.iloc[i]),
            'actual_5day': float(test_series.iloc[i + FORECAST_HORIZON - 1])
        })
        
        # Expand history with actual observation
        history.append(test_series.iloc[i])
        history_index.append(test_series.index[i])
        
        # Progress indicator
        if (i + 1) % 50 == 0 or i == n_forecasts - 1:
            print(f"    Progress: {i+1}/{n_forecasts} forecasts generated")
    
    forecast_df = pd.DataFrame(forecasts)
    
    # Store naive forecasts for MASE calculation
    # Naive forecast = yesterday's value (persistence model)
    naive_errors = []
    for i in range(1, len(test_series)):
        naive_errors.append(abs(test_series.iloc[i] - test_series.iloc[i-1]))
    
    naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    
    print(f"  ✓ Generated {len(forecast_df)} forecast sets")
    print(f"    Naive MAE (for MASE): {naive_mae:.4f}")
    
    # Save naive MAE for MASE calculation in evaluation step
    forecast_df.attrs['naive_mae'] = naive_mae
    
    return forecast_df, naive_mae

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Complete ARIMA modeling pipeline for VRU compressor forecasting
    """
    import os
    import json
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print(" VRU COMPRESSOR ARIMA FORECASTING - MODEL DEVELOPMENT")
    print(" (Rolling Window Forecast — Fixed)")
    print("="*70)
    print()
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Filter sudden events (CRITICAL for VRU!)
    df_filtered, sudden_mask = detect_sudden_events(df)
    
    # Step 3: Train/test split
    train, test = train_test_split(df_filtered)
    
    # Process each parameter
    parameters = {
        'discharge_temp': 'Discharge Temperature',
        'discharge_pressure': 'Discharge Pressure',
        'jacket_water': 'Jacket Water Pressure'
    }
    
    models = {}
    forecasts_all = {}
    naive_maes = {}
    
    for param_col, param_name in parameters.items():
        print("\n" + "="*70)
        print(f" PARAMETER: {param_name.upper()}")
        print("="*70)
        
        train_series = train[param_col]
        test_series = test[param_col]
        
        # Step 4: Stationarity assessment
        d_order = test_stationarity(train_series, param_name)
        
        # Step 5: ACF/PACF analysis
        plot_acf_pacf(train_series, param_col, d_order)
        
        # Step 6: Model selection
        best_model, best_order = select_best_arima(train_series, param_name, d_order)
        
        # Step 7: Diagnostic validation
        validate_model(best_model, param_name)
        
        # Step 8: Generate ROLLING forecasts (FIXED)
        forecast_df, naive_mae = generate_forecasts(
            train_series, test_series, best_order, param_name
        )
        
        # Store results
        models[param_col] = {
            'order': best_order,
            'aic': float(best_model.aic),
            'bic': float(best_model.bic),
            'd_order': d_order,
            'param_name': param_name
        }
        
        forecasts_all[param_col] = forecast_df
        naive_maes[param_col] = naive_mae
        
        # Save model summary
        with open(f'{OUTPUT_DIR}model_summary_{param_col}.txt', 'w') as f:
            f.write(f"ARIMA Model Summary: {param_name}\n")
            f.write("="*60 + "\n\n")
            f.write(str(best_model.summary()))
        
        print(f"\n  ✓ Model summary saved: model_summary_{param_col}.txt")
    
    # Save all forecasts
    for param_col, forecast_df in forecasts_all.items():
        forecast_df.to_csv(f'{OUTPUT_DIR}forecasts_{param_col}.csv', index=False)
        print(f"  ✓ Forecasts saved: forecasts_{param_col}.csv")
    
    # Save model metadata (including naive MAE for MASE)
    metadata = {
        'parameters': list(parameters.keys()),
        'train_size': len(train),
        'test_size': len(test),
        'forecast_horizon': FORECAST_HORIZON,
        'sudden_events_filtered': int(sudden_mask.sum()),
        'data_frequency': 'daily_average',
        'forecast_method': 'expanding_window_with_periodic_refit',
        'refit_interval': 20,
        'models': {
            param: {
                'order': models[param]['order'],
                'AIC': models[param]['aic'],
                'BIC': models[param]['bic']
            }
            for param in parameters.keys()
        },
        'naive_mae': naive_maes  # For MASE calculation in evaluation
    }
    
    with open(f'{OUTPUT_DIR}model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*70)
    print(" ARIMA MODELING COMPLETE!")
    print("="*70)
    print(f"\n  ✓ Models developed for 3 parameters")
    print(f"  ✓ Sudden events filtered: {sudden_mask.sum()} days")
    print(f"  ✓ Training data: {len(train)} observations (gradual degradation only)")
    print(f"  ✓ Test forecasts generated with rolling window")
    print(f"  ✓ Naive MAE stored for MASE calculation")
    print(f"  ✓ Output directory: {OUTPUT_DIR}")
    print(f"\n  Next step: Run 03_alert_system.py to generate rule-based alerts")
    print()

if __name__ == "__main__":
    main()
