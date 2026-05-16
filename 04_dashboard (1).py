import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="VRU Compressor Monitor",
    page_icon="⚙️",
    layout="wide",
)

st.markdown("""
<style>
/* ── page background ── */
.stApp { background-color: #ffffff; color: #111111; }
.main .block-container { background-color: #ffffff; padding-top: 1.5rem; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background-color: #f5f5f5;
    border-right: 1px solid #e0e0e0;
}
[data-testid="stSidebar"] * { color: #111111 !important; }

/* ── top bar ── */
header[data-testid="stHeader"] { background-color: #ffffff; border-bottom: 1px solid #e0e0e0; }
[data-testid="stToolbar"] { background-color: #ffffff; }

/* ── all text ── */
h1, h2, h3, h4, h5, h6, p, div, span, label { color: #111111; }
.stMarkdown, .stCaption { color: #333333; }

/* ── metric boxes ── */
[data-testid="metric-container"] {
    background-color: #f9f9f9;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="metric-container"] label { color: #555555 !important; font-size: 0.82em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #111111 !important; font-weight: 700; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #555555 !important; }

/* ── buttons & radio ── */
.stRadio label { color: #111111 !important; }
.stButton > button {
    background-color: #111111;
    color: #ffffff;
    border: none;
    border-radius: 6px;
}
.stButton > button:hover { background-color: #333333; }

/* ── inputs ── */
.stDateInput input, .stTextInput input, .stSelectbox select {
    background-color: #ffffff;
    color: #111111;
    border: 1px solid #cccccc;
    border-radius: 6px;
}

/* ── dataframe ── */
.stDataFrame { border: 1px solid #e0e0e0; border-radius: 6px; }
[data-testid="stDataFrame"] th {
    background-color: #f0f0f0 !important;
    color: #111111 !important;
    font-weight: 600;
}
[data-testid="stDataFrame"] td { color: #111111 !important; }

/* ── alerts (info/warning/error) ── */
.stAlert { border-radius: 6px; border-left-width: 4px; }
[data-baseweb="notification"] { background-color: #f9f9f9 !important; color: #111111 !important; }

/* ── horizontal divider ── */
hr { border-color: #e0e0e0; }

/* ── tab text ── */
.stTabs [data-baseweb="tab"] { color: #111111 !important; }
</style>
""", unsafe_allow_html=True)

# ── shared plotly layout ──────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='#ffffff',
    plot_bgcolor='#ffffff',
    font=dict(color='#111111', family='sans-serif'),
    xaxis=dict(showgrid=True, gridcolor='#eeeeee', linecolor='#cccccc', tickfont=dict(color='#333333')),
    yaxis=dict(showgrid=True, gridcolor='#eeeeee', linecolor='#cccccc', tickfont=dict(color='#333333')),
)

THRESHOLDS = {
    'discharge_temp':     {'NORMAL': (110, 150), 'WARNING': 150, 'CRITICAL': 300, 'OFF': 90},
    'discharge_pressure': {'NORMAL': (10, 30),   'LOW': 10,      'HIGH': 30},
    'jacket_water':       {'NORMAL': (12, 20),   'LOW': 12,      'HIGH': 20},
}

STATUS_COLORS = {
    'OFF':    '#9CA3AF',
    'GREEN':  '#16A34A',
    'YELLOW': '#D97706',
    'RED':    '#DC2626',
}

STATUS_BG = {
    'OFF':    '#F3F4F6',
    'GREEN':  '#DCFCE7',
    'YELLOW': '#FEF3C7',
    'RED':    '#FEE2E2',
}

# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data
def load_preprocessed():
    path = 'vru_preprocessed.csv'
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    else:
        df.index = pd.to_datetime(df.index)
        df.index.name = 'date'
    return df[['discharge_temp', 'discharge_pressure', 'jacket_water']]

@st.cache_data
def load_alerts():
    path = 'alerts/alerts_generated.csv'
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    df['target_date']   = pd.to_datetime(df['target_date'])
    return df

@st.cache_data
def load_forecasts():
    out = {}
    for param in ['discharge_temp', 'discharge_pressure', 'jacket_water']:
        path = f'arima_models/forecasts_{param}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['forecast_date'] = pd.to_datetime(df['forecast_date'])
            df['target_date']   = pd.to_datetime(df['target_date'])
            out[param] = df
    return out if out else None

@st.cache_data
def load_metadata():
    path = 'arima_models/model_metadata.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_failure_log():
    path = 'vru_failure_log.csv'
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['failure_date'] = pd.to_datetime(df['failure_date'])
    return df

# =============================================================================
# HELPERS
# =============================================================================

def status_badge(status):
    color = STATUS_COLORS.get(status, '#6B7280')
    bg    = STATUS_BG.get(status, '#F3F4F6')
    return (
        f'<span style="background:{bg};color:{color};padding:3px 12px;'
        f'border-radius:4px;font-weight:600;font-size:0.95em;'
        f'border:1px solid {color}33;">{status}</span>'
    )

def metric_card(label, value, unit, status_text, color):
    st.markdown(f"""
    <div style="background:#f9f9f9;border:1px solid #e0e0e0;border-radius:8px;
                padding:16px;text-align:center;">
        <div style="color:#555555;font-size:0.82em;margin-bottom:4px;">{label}</div>
        <div style="font-size:1.9em;font-weight:700;color:#111111;">
            {value} <span style="font-size:0.48em;color:#777777;">{unit}</span>
        </div>
        <div style="color:{color};font-size:0.83em;margin-top:4px;">{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

def apply_plot_theme(fig, **kwargs):
    layout = {**PLOT_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor='#eeeeee', linecolor='#cccccc')
    fig.update_yaxes(showgrid=True, gridcolor='#eeeeee', linecolor='#cccccc')
    return fig

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("⚙️ VRU Monitor")
st.sidebar.markdown("---")

page = st.sidebar.radio("Halaman", [
    "Overview",
    "Sensor History",
    "ARIMA Forecasts",
    "Alert Analysis",
    "Model Info",
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Threshold Reference**

| Parameter | Normal Range |
|---|---|
| Discharge Temp | 110–150 °F |
| Discharge Press | 10–30 psi |
| Jacket Water | 12–20 psi |
""")

# =============================================================================
# PAGE: OVERVIEW
# =============================================================================

if page == "Overview":
    st.title("VRU Compressor — Overview")
    st.markdown("---")

    alerts = load_alerts()
    pre    = load_preprocessed()
    meta   = load_metadata()

    if alerts is None or pre is None:
        st.error("Data belum tersedia. Jalankan dulu script 01, 02, dan 03.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Hari Operasional", f"{len(pre):,}")
    with col2:
        st.metric("Test Points", f"{len(alerts):,}")
    with col3:
        n_red    = (alerts['alert_actual'] == 'RED').sum()
        n_yellow = (alerts['alert_actual'] == 'YELLOW').sum()
        st.metric("Actual RED Events", n_red, delta=f"{n_yellow} YELLOW", delta_color="off")
    with col4:
        agreement = (alerts['alert_status'] == alerts['alert_actual']).mean() * 100
        st.metric("Forecast Accuracy", f"{agreement:.1f}%")

    st.markdown("---")
    st.subheader("Alert Distribution")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Forecast (5-day)**")
        for s in ['GREEN', 'YELLOW', 'RED', 'OFF']:
            n   = (alerts['alert_status'] == s).sum()
            pct = n / len(alerts) * 100
            st.markdown(
                f"{status_badge(s)} &nbsp; **{n}** hari &nbsp; ({pct:.1f}%)",
                unsafe_allow_html=True,
            )
            st.markdown("")

    with col_b:
        st.markdown("**Aktual (Ground Truth)**")
        for s in ['GREEN', 'YELLOW', 'RED', 'OFF']:
            n   = (alerts['alert_actual'] == s).sum()
            pct = n / len(alerts) * 100
            st.markdown(
                f"{status_badge(s)} &nbsp; **{n}** hari &nbsp; ({pct:.1f}%)",
                unsafe_allow_html=True,
            )
            st.markdown("")

    st.markdown("---")
    st.subheader("Missed Events")
    missed = alerts[
        alerts['alert_actual'].isin(['YELLOW', 'RED']) &
        (alerts['alert_status'] == 'GREEN')
    ][[
        'target_date', 'alert_actual',
        'temp_actual', 'press_actual', 'jacket_actual',
        'temp_forecast_5d', 'press_forecast_5d', 'jacket_forecast_5d',
    ]].copy()

    missed.columns = [
        'Target Date', 'Actual Status',
        'Actual Temp', 'Actual Press', 'Actual Jacket',
        'Forecast Temp', 'Forecast Press', 'Forecast Jacket',
    ]
    st.dataframe(missed, use_container_width=True)
    st.caption(
        f"Total missed: {len(missed)} hari. "
        "ARIMA forecast semua GREEN karena model belajar dari mean historis jangka panjang."
    )

# =============================================================================
# PAGE: SENSOR HISTORY
# =============================================================================

elif page == "Sensor History":
    st.title("Sensor History")
    pre = load_preprocessed()
    if pre is None:
        st.error("vru_preprocessed.csv tidak ditemukan.")
        st.stop()

    failures = load_failure_log()

    date_min = pre.index.min().date()
    date_max = pre.index.max().date()
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Dari", value=date_min, min_value=date_min, max_value=date_max)
    with col2:
        end   = st.date_input("Sampai", value=date_max, min_value=date_min, max_value=date_max)

    mask = (pre.index.date >= start) & (pre.index.date <= end)
    df   = pre[mask]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=[
            'Discharge Temperature (°F)',
            'Discharge Pressure (psi)',
            'Jacket Water Pressure (psi)',
        ],
    )

    fig.add_trace(go.Scatter(
        x=df.index, y=df['discharge_temp'], name='Discharge Temp',
        line=dict(color='#DC2626', width=1.2),
    ), row=1, col=1)
    fig.add_hline(y=150, line_dash='dash', line_color='#D97706', row=1, col=1)
    fig.add_hrect(y0=110, y1=150, fillcolor='#16A34A', opacity=0.05, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['discharge_pressure'], name='Discharge Pressure',
        line=dict(color='#2563EB', width=1.2),
    ), row=2, col=1)
    fig.add_hrect(y0=10, y1=30, fillcolor='#16A34A', opacity=0.05, row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['jacket_water'], name='Jacket Water',
        line=dict(color='#059669', width=1.2),
    ), row=3, col=1)
    fig.add_hrect(y0=12, y1=20, fillcolor='#16A34A', opacity=0.05, row=3, col=1)
    fig.add_hline(y=12, line_dash='dash', line_color='#DC2626', row=3, col=1)

    if failures is not None:
        for _, row in failures.iterrows():
            fd = row['failure_date']
            if pd.Timestamp(start) <= fd <= pd.Timestamp(end):
                for r in [1, 2, 3]:
                    fig.add_vline(x=fd, line_dash='dot', line_color='#DC2626', opacity=0.35, row=r, col=1)

    fig.update_layout(
        height=700, showlegend=False, margin=dict(t=60, b=40),
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        font=dict(color='#111111'),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#eeeeee', linecolor='#cccccc')
    fig.update_yaxes(showgrid=True, gridcolor='#eeeeee', linecolor='#cccccc')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Statistik Periode Dipilih")
    params = {
        'Discharge Temp (°F)':      'discharge_temp',
        'Discharge Pressure (psi)': 'discharge_pressure',
        'Jacket Water (psi)':       'jacket_water',
    }
    cols = st.columns(3)
    for (label, col), c in zip(params.items(), cols):
        s = df[col]
        with c:
            st.markdown(f"**{label}**")
            st.markdown(f"Mean: **{s.mean():.1f}** &nbsp;|&nbsp; Std: **{s.std():.2f}**", unsafe_allow_html=True)
            st.markdown(f"Min: **{s.min():.1f}** &nbsp;|&nbsp; Max: **{s.max():.1f}**", unsafe_allow_html=True)

# =============================================================================
# PAGE: ARIMA FORECASTS
# =============================================================================

elif page == "ARIMA Forecasts":
    st.title("ARIMA Forecasts vs Actual")
    forecasts = load_forecasts()
    meta      = load_metadata()

    if forecasts is None:
        st.error("Forecast data tidak ditemukan. Jalankan 02_arima_modeling.py dulu.")
        st.stop()

    param_labels = {
        'discharge_temp':     'Discharge Temperature (°F)',
        'discharge_pressure': 'Discharge Pressure (psi)',
        'jacket_water':       'Jacket Water Pressure (psi)',
    }

    if meta:
        st.subheader("Model yang Terpilih")
        col1, col2, col3 = st.columns(3)
        for (param, label), col in zip(param_labels.items(), [col1, col2, col3]):
            if param in meta.get('models', {}):
                m = meta['models'][param]
                with col:
                    st.markdown(f"**{label}**")
                    st.markdown(f"Model: `ARIMA{tuple(m['order'])}`")
                    st.markdown(f"AIC: {m['AIC']:.2f}")

    st.markdown("---")

    horizon = st.radio("Horizon", ["1-day", "5-day"], horizontal=True)
    fc_col  = 'forecast_1day' if horizon == "1-day" else 'forecast_5day'
    act_col = 'actual_1day'   if horizon == "1-day" else 'actual_5day'

    for param, label in param_labels.items():
        if param not in forecasts:
            continue
        df       = forecasts[param]
        mae      = np.mean(np.abs(df[act_col] - df[fc_col]))
        naive_mae = meta['naive_mae'][param] if meta and 'naive_mae' in meta else None
        mase     = mae / naive_mae if naive_mae else None

        st.subheader(label)
        c1, c2 = st.columns([3, 1])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['target_date'], y=df[act_col], name='Actual',
                line=dict(color='#111111', width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=df['target_date'], y=df[fc_col], name='Forecast',
                line=dict(color='#2563EB', width=1.5, dash='dash'),
            ))
            fig.update_layout(
                height=250, margin=dict(t=20, b=20),
                legend=dict(orientation='h', font=dict(color='#111111')),
                paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                font=dict(color='#111111'),
            )
            fig.update_xaxes(showgrid=True, gridcolor='#eeeeee', linecolor='#cccccc')
            fig.update_yaxes(showgrid=True, gridcolor='#eeeeee', linecolor='#cccccc')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.metric("MAE", f"{mae:.3f}")
            if mase is not None:
                delta_color = "normal" if mase < 1 else "inverse"
                st.metric(
                    "MASE", f"{mase:.3f}",
                    delta="beats naive" if mase < 1 else "below naive",
                    delta_color=delta_color,
                )

# =============================================================================
# PAGE: ALERT ANALYSIS
# =============================================================================

elif page == "Alert Analysis":
    st.title("Alert Analysis")
    alerts = load_alerts()
    if alerts is None:
        st.error("alerts/alerts_generated.csv tidak ditemukan.")
        st.stop()

    st.subheader("Alert Timeline")
    status_num = {'OFF': -1, 'GREEN': 0, 'YELLOW': 1, 'RED': 2}
    alerts['status_num']        = alerts['alert_status'].map(status_num)
    alerts['actual_status_num'] = alerts['alert_actual'].map(status_num)

    fig = go.Figure()
    for status, color in STATUS_COLORS.items():
        mask = alerts['alert_status'] == status
        if mask.any():
            fig.add_trace(go.Scatter(
                x=alerts.loc[mask, 'forecast_date'],
                y=alerts.loc[mask, 'status_num'],
                mode='markers',
                name=f'Forecast {status}',
                marker=dict(color=color, size=7, symbol='circle'),
            ))
    for status, color in STATUS_COLORS.items():
        mask = alerts['alert_actual'] == status
        if mask.any():
            fig.add_trace(go.Scatter(
                x=alerts.loc[mask, 'forecast_date'],
                y=[s + 0.3 for s in alerts.loc[mask, 'actual_status_num']],
                mode='markers',
                name=f'Actual {status}',
                marker=dict(color=color, size=7, symbol='diamond'),
            ))

    fig.update_layout(
        height=350,
        yaxis=dict(
            tickvals=[-1, 0, 1, 2],
            ticktext=['OFF', 'GREEN', 'YELLOW', 'RED'],
            tickfont=dict(color='#111111'),
        ),
        legend=dict(orientation='h', y=-0.3, font=dict(color='#111111')),
        margin=dict(t=20, b=80),
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        font=dict(color='#111111'),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#eeeeee', linecolor='#cccccc')
    fig.update_yaxes(showgrid=True, gridcolor='#eeeeee', linecolor='#cccccc')
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Lingkaran = forecast · Berlian = actual")

    st.markdown("---")
    st.subheader("Confusion Matrix")
    statuses = ['OFF', 'GREEN', 'YELLOW', 'RED']
    cm_data  = []
    for act in statuses:
        row = []
        for fc in statuses:
            row.append(((alerts['alert_actual'] == act) & (alerts['alert_status'] == fc)).sum())
        cm_data.append(row)

    cm_df = pd.DataFrame(
        cm_data,
        index=[f'Actual: {s}' for s in statuses],
        columns=[f'Forecast: {s}' for s in statuses],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Classification Metrics")
    from sklearn.metrics import classification_report
    report    = classification_report(
        alerts['alert_actual'], alerts['alert_status'],
        labels=['GREEN', 'YELLOW', 'RED'], output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.round(3), use_container_width=True)

    st.warning(
        "YELLOW recall = 0% dan RED recall = 0%. "
        "Semua YELLOW dan RED aktual diprediksi GREEN. "
        "Ini limitasi struktural ARIMA untuk anomaly detection berbasis threshold — "
        "bukan bug di kode. Dibahas di laporan review."
    )

# =============================================================================
# PAGE: MODEL INFO
# =============================================================================

elif page == "Model Info":
    st.title("Model Info")
    meta = load_metadata()

    if meta is None:
        st.error("arima_models/model_metadata.json tidak ditemukan.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Size", f"{meta['train_size']:,} obs")
    with col2:
        st.metric("Test Size", f"{meta['test_size']:,} obs")
    with col3:
        st.metric("Forecast Horizon", f"{meta['forecast_horizon']} hari")
    with col4:
        st.metric("Sudden Events Filtered", f"{meta['sudden_events_filtered']} hari")

    st.markdown("---")
    st.subheader("Model Orders & AIC")

    param_labels = {
        'discharge_temp':     'Discharge Temperature',
        'discharge_pressure': 'Discharge Pressure',
        'jacket_water':       'Jacket Water',
    }

    rows = []
    for param, label in param_labels.items():
        m         = meta['models'].get(param, {})
        naive_mae = meta.get('naive_mae', {}).get(param, None)
        rows.append({
            'Parameter': label,
            'Model':     f"ARIMA{tuple(m.get('order', ['?', '?', '?']))}",
            'AIC':       round(m.get('AIC', 0), 2),
            'BIC':       round(m.get('BIC', 0), 2),
            'Naive MAE': round(naive_mae, 4) if naive_mae else 'N/A',
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    st.subheader("Pipeline")
    st.code("""
vru_data_full_4years.csv
        ↓
01_data_preprocessing.py   → vru_preprocessed.csv
        ↓
02_arima_modeling.py       → arima_models/forecasts_*.csv
                              arima_models/model_metadata.json
        ↓
03_alert_system.py         → alerts/alerts_generated.csv
        ↓
04_dashboard.py            → dashboard ini
""", language=None)

    st.markdown("---")
    st.subheader("Catatan Limitasi")
    st.info("""
**Discharge Temp & Jacket Water MASE > 1.0**

Dua dari tiga parameter masih kalah dari naive forecast (tebak nilai kemarin).
Discharge pressure adalah satu-satunya parameter dengan MASE di bawah 1.0.

**Alert Classification Recall = 0% untuk YELLOW & RED**

ARIMA point forecast selalu jatuh di sekitar historical mean (143–146 °F untuk temp).
Ketika nilai aktual drift gradual mendekati threshold, model tidak sensitif menangkapnya.
Solusi yang bisa dieksplor: gunakan prediction interval bukan point forecast untuk classification,
atau tambahkan residual-based anomaly detection sebagai layer kedua.
""")
