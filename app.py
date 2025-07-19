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
import mplfinance as mpf
import warnings
import matplotlib.pyplot as plt

# Configuration
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")

# Constants
MIN_DATA_POINTS = 30
LOOKBACK_PERIOD = 5
DATA_RETRIEVAL_DAYS = 10
REQUIRED_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

st_autorefresh(interval=300000, key="data_refresh")
st.title("📈 SPY Options Forecast Dashboard")

# Sidebar controls
st.sidebar.header("🎛️ Strategy Tuner")

return_threshold = st.sidebar.slider(
    "Signal Threshold (log return)", 
    0.0005, 0.01, 0.0015, 0.0005,
    help="Minimum predicted log return required to consider a trade. Higher values filter out low-confidence, small-move trades.")

delta_assumption = st.sidebar.slider(
    "Option Delta", 
    0.1, 1.0, 0.5, 0.05,
    help="Delta represents sensitivity to the underlying price. Lower delta means cheaper contracts but more out-of-the-money risk.")

contract_cost = st.sidebar.number_input("Contract Cost ($)", 1.0, 1000.0, 1.5, 0.1)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

hold_mode = st.sidebar.radio(
    "Holding Strategy",
    ["Fixed Candles", "Until Signal Changes"],
    help="Choose whether to exit after a fixed number of bars or wait until the model gives a different signal.")

if hold_mode == "Fixed Candles":
    hold_period = st.sidebar.slider("Hold Period (candles)", 1, 6, 1, 1)
else:
    hold_period = 1  # Placeholder — actual hold duration handled in backtest logic


col1, col2 = st.columns(2)
with col1:
    show_ema = st.checkbox("Show EMA (9, 21, 50)", value=True)
    show_macd = st.checkbox("Show MACD", value=True)
with col2:
    show_rsi = st.checkbox("Show RSI", value=True)
    show_vwap = st.checkbox("Show VWAP", value=True)

def safe_download_data():
    try:
        with st.spinner("Downloading market data..."):
            df = yf.download("SPY", interval="15m", period=f"{DATA_RETRIEVAL_DAYS}d", prepost=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            if not isinstance(df, pd.DataFrame) or df.empty:
                st.warning("Initial download failed, trying fallback...")
                df = yf.Ticker("SPY").history(period=f"{DATA_RETRIEVAL_DAYS}d", interval="15m")
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
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
    if len(df) < MIN_DATA_POINTS:
        st.warning("Not enough valid OHLC data")
        return None
    df['Volume'] = df['Volume'].fillna(0)
    return df

def calculate_features(df):
    try:
        df = df.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        for w in [9, 21, 50]:
            df[f'EMA_{w}'] = EMAIndicator(close=close, window=w).ema_indicator()
        df['RSI'] = RSIIndicator(close=close).rsi()
        df['VWAP'] = VolumeWeightedAveragePrice(high, low, close, volume).volume_weighted_average_price()
        macd = MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        df['ATR'] = AverageTrueRange(high, low, close).average_true_range()
        df['Log_Return'] = np.log(close / close.shift(1))
        for i in range(1, LOOKBACK_PERIOD + 1):
            df[f'Return_lag_{i}'] = df['Log_Return'].shift(i)
        df['Target'] = (df['Log_Return'].shift(-1) > return_threshold).astype(int)
        features = ['Log_Return'] + [f'Return_lag_{i}' for i in range(1, LOOKBACK_PERIOD + 1)] + ['Target']
        df = df.dropna(subset=features)
        if len(df) < MIN_DATA_POINTS:
            st.warning("Too few samples after feature calculation")
            return None
        return df
    except Exception as e:
        st.error(f"Feature calculation failed: {str(e)}")
        return None

def backtest_trades(df, model, scaler, features, return_threshold, delta_assumption, contract_cost,
                    hold_mode="Fixed Candles", hold_period=1, confidence_threshold=0.0):
    trades = []
    df = df.copy()
    i = LOOKBACK_PERIOD

    while i < len(df) - 1:
        X_sample = df[features].iloc[i:i+1]
        X_scaled = scaler.transform(X_sample)
        proba = model.predict_proba(X_scaled)[0][1]

        signal = None
        if proba >= confidence_threshold:
            signal = "CALL"
            direction = 1
            probability = proba
        elif (1 - proba) >= confidence_threshold:
            signal = "PUT"
            direction = -1
            probability = 1 - proba
        else:
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


# === Pipeline ===
with st.status("Loading and processing data...", expanded=True) as status:
    raw_data = safe_download_data()
    cleaned_data = validate_and_clean_data(raw_data)
    processed_data = calculate_features(cleaned_data)
    status.update(label="Data processing complete!", state="complete")

if processed_data is None:
    st.stop()

features = [f'Return_lag_{i}' for i in range(1, LOOKBACK_PERIOD + 1)]
X = processed_data[features]
y = processed_data['Target']

with st.spinner("Training model and running backtest..."):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=5, class_weight='balanced')
    model.fit(X_scaled, y)
    bt_results = backtest_trades(
    processed_data,
    model,
    scaler,
    features,
    return_threshold,
    delta_assumption,
    contract_cost,
    hold_mode=hold_mode,
    hold_period=hold_period if hold_mode == "Fixed Candles" else 1,
    confidence_threshold=confidence_threshold)


# === Latest Prediction ===
try:
    latest_scaled = scaler.transform(processed_data[features].iloc[-1:].values)
    proba = model.predict_proba(latest_scaled)[0][1]
    expected_return_call = proba * delta_assumption * 100 - contract_cost
    expected_return_put = (1 - proba) * delta_assumption * 100 - contract_cost
    if expected_return_call > 0 and expected_return_call >= expected_return_put:
        signal = '📈 CALL'
        expected_return = expected_return_call
    elif expected_return_put > 0:
        signal = '📉 PUT'
        expected_return = expected_return_put
    else:
        signal = '🚫 NO TRADE'
        expected_return = 0
except Exception as e:
    st.error(f"Prediction failed: {str(e)}")
    st.stop()

st.subheader("🔍 Latest Prediction")
col1, col2, col3 = st.columns(3)
col1.metric("Signal", signal)
col2.metric("Confidence", f"{proba*100:.1f}%")
col3.metric("Expected Return", f"${expected_return:.2f}")

# === Backtest Results ===
st.subheader("💰 Backtest Results")
if bt_results.empty:
    st.write("No trades met the threshold.")
else:
    # Format timestamp for Excel and create download
    export_df = bt_results.copy()
    export_df['Time'] = export_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    csv_data = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Backtest CSV",
        data=csv_data,
        file_name="backtest_results.csv",
        mime='text/csv'
    )
    st.dataframe(bt_results.tail(10), use_container_width=True)
    total_pnl = bt_results["P&L"].sum()
    win_rate = (bt_results["P&L"] > 0).mean() * 100
    st.metric("Total P&L", f"${total_pnl:.2f}")
    st.metric("Win Rate", f"{win_rate:.1f}%")
    st.metric("Trades", f"{len(bt_results)}")
    
    st.line_chart(bt_results["P&L"].cumsum(), use_container_width=True)

    # Confidence bucket breakdown
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

    

    # Equity vs SPY
    equity = bt_results["P&L"].cumsum()
    equity.index = bt_results["Time"]
    price = processed_data["Close"].loc[equity.index.min():equity.index.max()]
    st.line_chart(pd.DataFrame({"Equity Curve": equity, "SPY Price": price}))

# === Chart ===
st.subheader("📊 Technical Analysis")
plot_data = processed_data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-100:]
apds = []
if show_ema:
    for w in [9, 21, 50]:
        apds.append(mpf.make_addplot(processed_data[f'EMA_{w}'].iloc[-100:], color='blue', width=0.8))
if show_vwap:
    apds.append(mpf.make_addplot(processed_data['VWAP'].iloc[-100:], color='purple', width=0.8))
if show_macd:
    apds.append(mpf.make_addplot(processed_data['MACD'].iloc[-100:], panel=1, color='green'))
    apds.append(mpf.make_addplot(processed_data['MACD_Signal'].iloc[-100:], panel=1, color='orange'))
if show_rsi:
    apds.append(mpf.make_addplot(processed_data['RSI'].iloc[-100:], panel=2, color='black'))
processed_data['Signal_Marker'] = np.nan
processed_data.loc[processed_data.index[-1], 'Signal_Marker'] = processed_data['Close'].iloc[-1]
apds.append(mpf.make_addplot(processed_data['Signal_Marker'].iloc[-100:], type='scatter', markersize=120, marker='*', color='red'))

try:
    fig, _ = mpf.plot(
        plot_data,
        type='candle',
        addplot=apds,
        volume=True,
        panel_ratios=(4, 1, 1),
        style='yahoo',
        returnfig=True
    )
    st.pyplot(fig)
except Exception as e:
    st.error(f"Chart rendering failed: {str(e)}")





