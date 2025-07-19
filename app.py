# =============================================================================
# SPY Options Forecast Dashboard
# -----------------------------------------------------------------------------
# Author: Chris Schallberger
# Created: July 2025
# Description:
#   This Streamlit dashboard forecasts intraday SPY price movement using a 
#   machine learning model (RandomForestClassifier). It provides signals for 
#   CALL or PUT options based on log return thresholds and confidence levels.
#   The app includes backtesting with options-style P&L calculations, 
#   adjustable strategy parameters, technical analysis overlays, and 
#   visual performance tracking.
#
#   Features:
#     - Real-time 15-minute interval SPY data from Yahoo Finance
#     - Technical indicators: EMA, VWAP, RSI, MACD, ATR
#     - Strategy tuner for delta, return threshold, contract cost
#     - Confidence-filtered signal prediction
#     - Option to hold trades for fixed candles or until signal changes
#     - Backtesting with P&L tracking, equity curve, and confidence buckets
#     - Downloadable trade history with Excel-compatible timestamps
#
# License: All rights reserved. See LICENSE file for terms of use.
# =============================================================================


import streamlit as st
from streamlit_autorefresh import st_autorefresh
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.volatility import AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import mplfinance as mpf
import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go




warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")

# === Constants ===
MIN_DATA_POINTS = 30
LOOKBACK_PERIOD = 5
REQUIRED_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

st_autorefresh(interval=300000, key="data_refresh")
st.title("📈 SPY Options Forecast Dashboard")

# === Sidebar ===
st.sidebar.header("🕒 Data Settings")
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "60m", "90m", "1d"], index=2)
period = st.sidebar.selectbox("History Length", ["1d", "2d", "5d", "10d", "1mo", "3mo"], index=3)

if interval == "1m" and period not in ["1d", "2d", "5d", "7d"]:
    st.warning("⚠️ Yahoo Finance only supports up to 7 days for 1-minute interval.")

st.sidebar.header("🎛️ Strategy Tuner")
return_threshold = st.sidebar.slider("Signal Threshold (log return)", 0.0005, 0.01, 0.0015, 0.0005)
delta_assumption = st.sidebar.slider("Option Delta", 0.1, 1.0, 0.5, 0.05)
contract_cost = st.sidebar.number_input("Contract Cost ($)", 1.0, 1000.0, 1.5, 0.1)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
confidence_mode = st.sidebar.radio(
    "Confidence Filter Mode",
    ["Exact Threshold Only", "Include All Above Threshold"],
    help="Controls whether to include only predictions near the threshold, or all stronger signals too.")
hold_mode = st.sidebar.radio("Holding Strategy", ["Fixed Candles", "Until Signal Changes"])
hold_period = st.sidebar.slider("Hold Period (candles)", 1, 6, 1, 1) if hold_mode == "Fixed Candles" else 1

# === Indicator Config ===
TECHNICAL_INDICATORS = [
    {
        "name": "EMA 9",
        "key": "ema9",
        "chart": True,
        "function": lambda df: df.assign(EMA_9=EMAIndicator(df['Close'], window=9).ema_indicator())
    },
    {
        "name": "EMA 21",
        "key": "ema21",
        "chart": True,
        "function": lambda df: df.assign(EMA_21=EMAIndicator(df['Close'], window=21).ema_indicator())
    },
    {
        "name": "EMA 50",
        "key": "ema50",
        "chart": True,
        "function": lambda df: df.assign(EMA_50=EMAIndicator(df['Close'], window=50).ema_indicator())
    },
    {
        "name": "MACD",
        "key": "macd",
        "chart": True,
        "function": lambda df: df.assign(
            MACD=MACD(df['Close']).macd(),
            MACD_Signal=MACD(df['Close']).macd_signal(),
            MACD_Hist=MACD(df['Close']).macd_diff()
        )
    },
    {
        "name": "RSI",
        "key": "rsi",
        "chart": True,
        "function": lambda df: df.assign(RSI=RSIIndicator(df['Close']).rsi())
    },
    {
        "name": "VWAP",
        "key": "vwap",
        "chart": True,
        "function": lambda df: df.assign(
            VWAP=VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
        )
    },
    {
        "name": "ATR",
        "key": "atr",
        "chart": False,
        "function": lambda df: df.assign(
            ATR=AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        )
    },
    {
        "name": "Stochastic Oscillator",
        "key": "stoch",
        "chart": True,
        "function": lambda df: df.assign(
            Stoch_K=((df['Close'] - df['Low'].rolling(14).min()) /
                     (df['High'].rolling(14).max() - df['Low'].rolling(14).min()) * 100),
            Stoch_D=lambda x: x['Stoch_K'].rolling(3).mean()
        )
    },
    {
        "name": "CCI (Commodity Channel Index)",
        "key": "cci",
        "chart": True,
        "function": lambda df: df.assign(
            CCI=(df['Close'] - (df['High'] + df['Low'] + df['Close']) / 3).rolling(20).apply(
                lambda x: (x[-1] - x.mean()) / (0.015 * x.std()) if x.std() != 0 else 0, raw=False)
        )
    },
    {
        "name": "ADX (Average Directional Index)",
        "key": "adx",
        "chart": False,
        "function": lambda df: df.assign(
            ADX=AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range().rolling(14).mean()
        )
    },
    {
        "name": "Bollinger Band Width",
        "key": "bbwidth",
        "chart": True,
        "function": lambda df: df.assign(
            BB_MID=df['Close'].rolling(20).mean(),
            BB_STD=df['Close'].rolling(20).std(),
            BB_WIDTH=(df['Close'].rolling(20).std() * 4) / df['Close'].rolling(20).mean()
        )
    },
    {
        "name": "Momentum (Rate of Change)",
        "key": "mom",
        "chart": True,
        "function": lambda df: df.assign(
            MOM=df['Close'].pct_change(periods=5) * 100
        )
    },
    {
        "name": "Donchian Channel",
        "key": "donchian",
        "chart": True,
        "function": lambda df: df.assign(
            Donchian_Upper=df['High'].rolling(20).max(),
            Donchian_Lower=df['Low'].rolling(20).min()
        )
    }
]


st.sidebar.header("📊 Technical Indicators")
indicator_flags = {ind["key"]: st.sidebar.checkbox(ind["name"], value=True) for ind in TECHNICAL_INDICATORS}

# === Data Functions ===
def safe_download_data(interval, period):
    try:
        with st.spinner("Downloading market data..."):
            df = yf.download("SPY", interval=interval, period=period, prepost=True, progress=False)
            df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
            if df.empty:
                df = yf.Ticker("SPY").history(interval=interval, period=period)
                df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
            return df
    except Exception as e:
        st.error(f"Data download failed: {str(e)}")
        return None

def validate_and_clean_data(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.error("Invalid data format")
        return None
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return None
    df = df.copy()
    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[df[['Open', 'High', 'Low', 'Close']].notna().all(axis=1)]
    df['Volume'] = df['Volume'].fillna(0)
    if len(df) < MIN_DATA_POINTS:
        st.warning("Not enough valid OHLC data")
        return None
    return df

def calculate_features(df):
    try:
        df = df.copy()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        for i in range(1, LOOKBACK_PERIOD + 1):
            df[f'Return_lag_{i}'] = df['Log_Return'].shift(i)
        for ind in TECHNICAL_INDICATORS:
            if indicator_flags[ind["key"]]:
                df = ind["function"](df)
        df['Target'] = (df['Log_Return'].shift(-1) > return_threshold).astype(int)
        feature_cols = [col for col in df.columns if col.startswith("Return_lag_") or col == "Log_Return"]
        for ind in TECHNICAL_INDICATORS:
            if indicator_flags[ind["key"]]:
                for col in df.columns:
                    if ind["key"].upper() in col or col.startswith(ind["key"].upper()):
                        feature_cols.append(col)
        df = df.dropna(subset=feature_cols + ['Target'])
        if len(df) < MIN_DATA_POINTS:
            st.warning("Too few samples after feature calculation")
            return None, None
        return df, feature_cols
    except Exception as e:
        st.error(f"Feature calculation failed: {str(e)}")
        return None, None

def backtest_trades(df, model, scaler, features, return_threshold, delta_assumption, contract_cost,
                    hold_mode="Fixed Candles", hold_period=1, confidence_threshold=0.0,
                    confidence_mode="Include All Above Threshold"):
    trades = []
    df = df.copy()
    i = LOOKBACK_PERIOD

    while i < len(df) - 1:
        X_sample = df[features].iloc[i:i+1]
        X_scaled = scaler.transform(X_sample)
        proba = model.predict_proba(X_scaled)[0][1]

        signal = None
        direction = 0
        probability = 0
        match = False

        # === Corrected logic ===
        if confidence_mode.strip().lower() == "include all above threshold":
            if proba >= confidence_threshold:
                signal = "CALL"
                direction = 1
                probability = proba
                match = True
            elif (1 - proba) >= confidence_threshold:
                signal = "PUT"
                direction = -1
                probability = 1 - proba
                match = True
        else:  # Exact Threshold Only (±0.01 band)
            if abs(proba - confidence_threshold) < 0.01:
                signal = "CALL" if proba > 0.5 else "PUT"
                direction = 1 if signal == "CALL" else -1
                probability = proba if signal == "CALL" else 1 - proba
                match = True

        if not match:
            i += 1
            continue

        entry_price = df['Close'].iloc[i]

        if hold_mode == "Fixed Candles":
            exit_index = min(i + hold_period, len(df) - 1)
        else:
            exit_index = i + 1
            while exit_index < len(df):
                X_next = df[features].iloc[exit_index:exit_index+1]
                next_scaled = scaler.transform(X_next)
                next_proba = model.predict_proba(next_scaled)[0][1]

                if direction == 1 and next_proba < confidence_threshold:
                    break
                if direction == -1 and (1 - next_proba) < confidence_threshold:
                    break

                exit_index += 1

            if exit_index >= len(df):
                break

        exit_price = df['Close'].iloc[exit_index]
        pnl = ((exit_price - entry_price) * direction * delta_assumption * 100) - contract_cost

        trades.append({
            "Time": df.index[i],
            "Signal": signal,
            "Probability": probability,
            "Entry": entry_price,
            "Exit": exit_price,
            "Hold Bars": exit_index - i,
            "P&L": pnl
        })

        i = exit_index  # move forward to avoid overlapping trades

    return pd.DataFrame(trades)


# === Main Pipeline ===
with st.status("Loading and processing data...", expanded=True) as status:
    raw_data = safe_download_data(interval, period)
    cleaned_data = validate_and_clean_data(raw_data)
    processed_data, features = calculate_features(cleaned_data)
    status.update(label="Data processing complete!", state="complete")

if processed_data is None or features is None:
    st.stop()

X, y = processed_data[features], processed_data['Target']
with st.spinner("Training model and running backtest..."):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    base_model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=5, class_weight='balanced')
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    model.fit(X_scaled, y)
    bt_results = backtest_trades(processed_data, model, scaler, features,
                                 return_threshold, delta_assumption, contract_cost,
                                 hold_mode, hold_period, confidence_threshold, confidence_mode)

# === Prediction ===
try:
    latest_scaled = scaler.transform(processed_data[features].iloc[-1:].values)
    proba = model.predict_proba(latest_scaled)[0][1]
    expected_return_call = proba * delta_assumption * 100 - contract_cost
    expected_return_put = (1 - proba) * delta_assumption * 100 - contract_cost
    if expected_return_call > 0 and expected_return_call >= expected_return_put:
        signal, expected_return = '📈 CALL', expected_return_call
    elif expected_return_put > 0:
        signal, expected_return = '📉 PUT', expected_return_put
    else:
        signal, expected_return = '🚫 NO TRADE', 0
except Exception as e:
    st.error(f"Prediction failed: {str(e)}")
    st.stop()

st.subheader("🔍 Latest Prediction")
col1, col2, col3 = st.columns(3)
col1.metric("Signal", signal)
col2.metric("Confidence", f"{proba*100:.1f}%")
col3.metric("Expected Return", f"${expected_return:.2f}")

# === Backtest Display ===
st.subheader("💰 Backtest Results")
if bt_results.empty:
    st.write("No trades met the threshold.")
else:
    export_df = bt_results.copy()
    export_df['Time'] = export_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    st.download_button("📥 Download Full Backtest CSV", data=export_df.to_csv(index=False).encode('utf-8'),
                       file_name="backtest_results.csv", mime='text/csv')
    st.dataframe(bt_results.tail(10), use_container_width=True)
    st.metric("Total P&L", f"${bt_results['P&L'].sum():.2f}")
    st.metric("Win Rate", f"{(bt_results['P&L'] > 0).mean()*100:.1f}%")
    st.metric("Trades", f"{len(bt_results)}")
    st.line_chart(bt_results["P&L"].cumsum(), use_container_width=True)
    bt_results['Confidence_Bucket'] = (bt_results['Probability'] * 10).astype(int) / 10.0
    bucket_stats = bt_results.groupby('Confidence_Bucket').agg(
        Trades=('P&L', 'count'),
        Wins=('P&L', lambda x: (x > 0).sum()),
        WinRate=('P&L', lambda x: (x > 0).mean() * 100),
        AvgPnL=('P&L', 'mean')
    ).reset_index()
    st.subheader("📊 Performance by Confidence Bucket")
    st.dataframe(bucket_stats)
    fig, ax = plt.subplots()
    ax.bar(bucket_stats['Confidence_Bucket'], bucket_stats['WinRate'], width=0.05)
    ax.set_xlabel('Confidence Bucket')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rate by Confidence')
    ax.set_ylim(0, 100)
    st.pyplot(fig)
    equity = bt_results["P&L"].cumsum()
    equity.index = bt_results["Time"]
    price = processed_data["Close"].loc[equity.index.min():equity.index.max()]
    st.line_chart(pd.DataFrame({"Equity Curve": equity, "SPY Price": price}))

# === Chart ===
# === Price/Volume Section ===
st.subheader("📊 Price/Volume (Zoomable)")

# Use same winsorization on OHLC
def winsorize_series(series, lower=0.01, upper=0.99):
    q_low = series.quantile(lower)
    q_high = series.quantile(upper)
    return series.clip(lower=q_low, upper=q_high)

# Slice and clean last 100 rows
last_100 = processed_data.iloc[-100:].copy()
for col in ['Open', 'High', 'Low', 'Close']:
    last_100[col] = winsorize_series(last_100[col])

# Create candlestick + volume chart
fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(
    x=last_100.index,
    open=last_100['Open'],
    high=last_100['High'],
    low=last_100['Low'],
    close=last_100['Close'],
    name='Price'
))

# Volume as bar trace
fig.add_trace(go.Bar(
    x=last_100.index,
    y=last_100['Volume'],
    name='Volume',
    marker_color='lightgray',
    yaxis='y2',
    opacity=0.4
))

# Add large red star for signal marker
fig.add_trace(go.Scatter(
    x=[last_100.index[-1]],
    y=[last_100['Close'].iloc[-1]],
    mode='markers',
    marker=dict(symbol='star', size=16, color='red'),
    name='Signal'
))

# Layout: dual y-axes
fig.update_layout(
    title='Price & Volume (Zoom Enabled)',
    xaxis=dict(type='category'),
    yaxis=dict(title='Price'),
    yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    height=600,
    margin=dict(t=40, b=40)
)

st.plotly_chart(fig, use_container_width=True)





