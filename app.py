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
hold_mode = st.sidebar.radio("Holding Strategy", ["Fixed Candles", "Until Signal Changes"])
hold_period = st.sidebar.slider("Hold Period (candles)", 1, 6, 1, 1) if hold_mode == "Fixed Candles" else 1

# === Indicator Config ===
TECHNICAL_INDICATORS = [
    {
        "name": "EMA (9, 21, 50)",
        "key": "ema",
        "chart": True,
        "function": lambda df: df.assign(
            EMA_9=EMAIndicator(df['Close'], window=9).ema_indicator(),
            EMA_21=EMAIndicator(df['Close'], window=21).ema_indicator(),
            EMA_50=EMAIndicator(df['Close'], window=50).ema_indicator()
        )
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
        "function": lambda df: df.assign(VWAP=VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price())
    },
    {
        "name": "ATR",
        "key": "atr",
        "chart": False,
        "function": lambda df: df.assign(ATR=AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range())
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
                    hold_mode="Fixed Candles", hold_period=1, confidence_threshold=0.0):
    trades = []
    df = df.copy()
    i = LOOKBACK_PERIOD
    while i < len(df) - 1:
        X_sample = df[features].iloc[i:i+1]
        X_scaled = scaler.transform(X_sample)
        proba = model.predict_proba(X_scaled)[0][1]
        signal, direction, probability = None, 0, 0
        if proba >= confidence_threshold:
            signal, direction, probability = "CALL", 1, proba
        elif (1 - proba) >= confidence_threshold:
            signal, direction, probability = "PUT", -1, 1 - proba
        else:
            i += 1
            continue
        entry_price = df['Close'].iloc[i]
        exit_index = min(i + hold_period, len(df) - 1) if hold_mode == "Fixed Candles" else i + 1
        while hold_mode != "Fixed Candles" and exit_index < len(df):
            next_scaled = scaler.transform(df[features].iloc[exit_index:exit_index+1])
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
        i = exit_index
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
                                 hold_mode, hold_period, confidence_threshold)

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
st.subheader("📊 Technical Analysis")
plot_data = processed_data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-100:]
apds = []
for ind in TECHNICAL_INDICATORS:
    if indicator_flags[ind["key"]] and ind["chart"]:
        if ind["key"] == "ema":
            for w in [9, 21, 50]:
                apds.append(mpf.make_addplot(processed_data[f'EMA_{w}'].iloc[-100:], color='blue'))
        elif ind["key"] == "macd":
            apds.append(mpf.make_addplot(processed_data['MACD'].iloc[-100:], panel=1, color='green'))
            apds.append(mpf.make_addplot(processed_data['MACD_Signal'].iloc[-100:], panel=1, color='orange'))
        elif ind["key"] == "rsi":
            apds.append(mpf.make_addplot(processed_data['RSI'].iloc[-100:], panel=2, color='black'))
        elif ind["key"] == "vwap":
            apds.append(mpf.make_addplot(processed_data['VWAP'].iloc[-100:], color='purple'))

processed_data['Signal_Marker'] = np.nan
processed_data.loc[processed_data.index[-1], 'Signal_Marker'] = processed_data['Close'].iloc[-1]
apds.append(mpf.make_addplot(processed_data['Signal_Marker'].iloc[-100:], type='scatter', markersize=120, marker='*', color='red'))

try:
    fig, _ = mpf.plot(plot_data, type='candle', addplot=apds, volume=True,
                      panel_ratios=(4, 1, 1), style='yahoo', returnfig=True)
    st.pyplot(fig)
except Exception as e:
    st.error(f"Chart rendering failed: {str(e)}")





