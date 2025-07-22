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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import mplfinance as mpf
import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tzlocal import get_localzone
from pytz import timezone
from datetime import time
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



# ✅ Convert to local time and filter to market hours
eastern = timezone("US/Eastern")
market_open = time(9, 30)
market_close = time(16, 0)

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")

# === Constants ===
MIN_DATA_POINTS = 30
LOOKBACK_PERIOD = 5
REQUIRED_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

# === Ticker Selection ===
st.sidebar.header("💡 Ticker Selection")
selected_ticker = st.sidebar.text_input("Enter Ticker Symbol", value="SPY").upper()

st_autorefresh(interval=300000, key="data_refresh")
st.title(f"📈 {selected_ticker} Options Forecast Dashboard")


# === Sidebar ===

st.sidebar.header("🧠 Model Selection")


model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "Gradient Boosting",
        "Random Forest",
        "Hist Gradient Boosting",
        "Logistic Regression",
        "Support Vector Machine",
        "K-Nearest Neighbors",
        "Naive Bayes"
    ],
    index=0
)

st.sidebar.subheader("🛠 Model Hyperparameters")

# Default values to prevent undefined errors
n_estimators, max_depth, learning_rate, n_neighbors = 100, 5, 0.05, 5

if model_name in ["Gradient Boosting", "Random Forest", "Hist Gradient Boosting"]:
    n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 150, step=10)
    max_depth = st.sidebar.slider("Max Tree Depth", 2, 20, 4)

if model_name in ["Gradient Boosting", "Hist Gradient Boosting"]:
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.05, step=0.01)

if model_name == "K-Nearest Neighbors":
    n_neighbors = st.sidebar.slider("K (Neighbors)", 1, 20, 5)




st.sidebar.header("🕒 Data Settings")
interval = st.sidebar.selectbox(
    "Interval",
    ["1m", "5m", "15m", "30m", "60m", "90m", "1d"],
    index=2,
    help="Select the chart interval (1m to 1d)."
)
period = st.sidebar.selectbox(
    "History Length",
    ["1d", "2d", "5d", "10d", "1mo", "3mo"],
    index=3,
    help="Choose how much historical data to load."
)
limit_market_hours = st.sidebar.checkbox("Limit Backtest to Market Hours", value=True)

st.sidebar.header("🎛️ Strategy Tuner")
return_threshold = st.sidebar.slider(
    "Signal Threshold (log return)",
    min_value=0.0001,
    max_value=0.01,
    value=0.0015,
    step=0.0001,
    format="%.4f",
    help="Minimum expected log return to trigger a signal (e.g., 0.0010 = 0.10%)."
)

delta_assumption = st.sidebar.slider(
    "Option Delta",
    0.1, 1.0, 0.5, 0.05,
    help="Used to estimate how much the option moves per $1 move in {selected_ticker}."
)
contract_cost = st.sidebar.number_input(
    "Contract Cost ($)",
    1.0, 1000.0, 1.5, 0.1,
    help="Estimated cost of entering one options contract."
)
confidence_threshold = st.sidebar.slider(
    "📈 Confidence Threshold",
    min_value=0.25,
    max_value=1.0,
    value=0.5,
    step=0.0001,
    help="Minimum confidence required to enter a trade. Ranges from 0.25 (lenient) to 1.0 (very strict)."
)
confidence_mode = st.sidebar.radio(
    "Confidence Filter Mode",
    ["Exact Threshold Only", "Include All Above Threshold"],
    index=1,
    help="Exact: triggers when probability is near threshold. Include All: triggers for all higher-confidence signals."
)
hold_mode = st.sidebar.radio(
    "Holding Strategy",
    ["Fixed Candles", "Until Signal Changes"],
    index=1,  # <-- sets "Until Signal Changes" as default
    help="Fixed: exit after set number of candles. Signal Changes: exit when prediction direction flips."
)
hold_period = st.sidebar.slider(
    "Hold Period (candles)", 1, 6, 1, 1,
    help="Number of candles to hold a position.",
) if hold_mode == "Fixed Candles" else 1

# === Technical Indicator Toggles ===
st.sidebar.header("📊 Technical Indicators")
indicator_flags = {
    "ema9": st.sidebar.checkbox("EMA 9", value=False, help="9-period Exponential Moving Average."),
    "ema21": st.sidebar.checkbox("EMA 21", value=False, help="21-period EMA."),
    "ema50": st.sidebar.checkbox("EMA 50", value=False, help="50-period EMA."),
    "macd": st.sidebar.checkbox("MACD", value=False, help="MACD: Measures momentum and trend strength."),
    "rsi": st.sidebar.checkbox("RSI", value=True, help="RSI: Relative Strength Index — overbought/oversold levels."),
    "vwap": st.sidebar.checkbox("VWAP", value=False, help="VWAP: Average price weighted by volume."),
    "atr": st.sidebar.checkbox("ATR", value=False, help="ATR: Measures market volatility."),
    "stoch": st.sidebar.checkbox("Stochastic Oscillator", value=False, help="Stochastic momentum vs recent range."),
    "cci": st.sidebar.checkbox("CCI", value=False, help="CCI: Detect cyclical trends."),
    "adx": st.sidebar.checkbox("ADX", value=False, help="ADX: Strength of current trend."),
    "bbwidth": st.sidebar.checkbox("Bollinger Band Width", value=False, help="Measures band width for volatility."),
    "mom": st.sidebar.checkbox("Momentum (Rate of Change)", value=False, help="Rate of price change over time."),
    "donchian": st.sidebar.checkbox("Donchian Channel", value=False, help="Shows breakout boundaries.")
}


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


#st.sidebar.header("📊 Technical Indicators")
#indicator_flags = {ind["key"]: st.sidebar.checkbox(ind["name"], value=True) for ind in TECHNICAL_INDICATORS}

# === Data Functions ===
def safe_download_data(interval, period):
    try:
        with st.spinner("Downloading market data..."):
            df = yf.download(selected_ticker, interval=interval, period=period, prepost=True, progress=False)
            df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
            if df.empty:
                df = yf.Ticker(selected_ticker).history(interval=interval, period=period)
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

        # Return lags
        for i in range(1, LOOKBACK_PERIOD + 1):
            df[f'Return_lag_{i}'] = df['Log_Return'].shift(i)

        # === New engineered features ===
        df['Volatility_3'] = df['Log_Return'].rolling(3).std()
        df['Volatility_10'] = df['Log_Return'].rolling(10).std()
        df['Return_3_to_10'] = (
            df['Log_Return'].rolling(3).mean() /
            df['Log_Return'].rolling(10).mean()
        )
        df['Hour'] = df.index.hour + df.index.minute / 60.0


        # Apply selected indicators
        for ind in TECHNICAL_INDICATORS:
            if indicator_flags[ind["key"]]:
                df = ind["function"](df)

        # ✅ Add rule-based context features
        if 'EMA_50' in df.columns:
            df['EMA50_Uptrend'] = (df['Close'] > df['EMA_50']).astype(int)

        if 'RSI' in df.columns:
            df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
            df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)

        if 'MACD' in df.columns and 'MACD_Signal' in df.columns and 'MACD_Hist' in df.columns:
            df['MACD_Crossover'] = (df['MACD'] > df['MACD_Signal']).astype(int)
            df['MACD_Hist_Positive'] = (df['MACD_Hist'] > 0).astype(int)

        if 'VWAP' in df.columns:
            df['Price_Above_VWAP'] = (df['Close'] > df['VWAP']).astype(int)

        # Define binary classification target
        df['Target'] = (df['Log_Return'].shift(-1) > return_threshold).astype(int)
        st.write("📊 Target Label Distribution (Before Dropna):", df['Target'].value_counts())

        # Step 1: Start with return-based features
        feature_cols = [col for col in df.columns if col.startswith("Return_lag_") or col == "Log_Return"]
        feature_cols += ['Volatility_3', 'Volatility_10', 'Return_3_to_10', 'Hour']


        # Step 2: Add indicator-derived features
        for ind in TECHNICAL_INDICATORS:
            if indicator_flags[ind["key"]]:
                for col in df.columns:
                    if ind["key"].upper() in col.upper() or col.lower().startswith(ind["key"]):
                        if col not in feature_cols:
                            feature_cols.append(col)

        # Step 3: Add rule-based feature names if present
        rule_features = [
            "EMA50_Uptrend",
            "RSI_Overbought",
            "RSI_Oversold",
            "MACD_Crossover",
            "MACD_Hist_Positive",
            "Price_Above_VWAP"
        ]
        for f in rule_features:
            if f in df.columns and f not in feature_cols:
                feature_cols.append(f)

        # Drop NaN rows before training
        df = df.dropna(subset=feature_cols + ['Target'])
        if len(df) < MIN_DATA_POINTS:
            st.warning("Too few samples after feature calculation")
            return None, None

        return df, feature_cols

    except Exception as e:
        st.error(f"Feature calculation failed: {str(e)}")
        return None, None

import pandas as pd
import numpy as np

def backtest_trades(
    df, model, scaler, features,
    return_threshold, delta_assumption, contract_cost,
    hold_mode="Fixed Candles", hold_period=1,
    confidence_threshold=0.0, confidence_mode="Include All Above Threshold",
    slippage=0.02, commission=1.0, max_hold_bars=10, position_size=1
):

    trades = []
    df = df.copy()
    i = 5  # LOOKBACK_PERIOD constant

    equity = 0
    peak_equity = 0

    while i < len(df) - 1:
        X_sample = df[features].iloc[i:i + 1]
        X_scaled = scaler.transform(X_sample)
        proba = model.predict_proba(X_scaled)[0][1]

        signal, direction, probability = None, 0, 0
        match = False
        conf_mode = confidence_mode.strip().lower()

        if conf_mode == "include all above threshold":
            if proba >= 0.5 and proba >= confidence_threshold:
                signal, direction, probability, match = "CALL", 1, proba, True
            elif proba < 0.5 and (1 - proba) >= confidence_threshold:
                signal, direction, probability, match = "PUT", -1, 1 - proba, True

        elif conf_mode == "exact threshold only" and abs(proba - confidence_threshold) < 0.01:
            signal = "CALL" if proba > 0.5 else "PUT"
            direction = 1 if proba > 0.5 else -1
            probability = proba if proba > 0.5 else 1 - proba
            match = True

        if not match:
            i += 1
            continue

        entry_price = df['Close'].iloc[i] + (slippage * direction)

        if hold_mode == "Fixed Candles":
            exit_index = min(i + hold_period, len(df) - 1)
        else:
            exit_index = i + 1
            bars_held = 0
            while exit_index < len(df) and bars_held < max_hold_bars:
                X_next = df[features].iloc[exit_index:exit_index + 1]
                next_scaled = scaler.transform(X_next)
                next_proba = model.predict_proba(next_scaled)[0][1]
                if direction == 1 and next_proba < confidence_threshold:
                    break
                if direction == -1 and (1 - next_proba) < confidence_threshold:
                    break
                exit_index += 1
                bars_held += 1

            if exit_index >= len(df):
                break

        exit_price = df['Close'].iloc[exit_index] - (slippage * direction)
        raw_return = ((exit_price - entry_price) * direction * delta_assumption * 100)
        pnl = (raw_return - commission) * position_size
        equity += pnl
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity

        trades.append({
        "Time": df.index[i],
        "Signal": signal,
        "Probability": probability,
        "Entry": entry_price,
        "Exit": exit_price,
        "Hold Bars": exit_index - i,
        "P&L": pnl,
        "Equity": equity,
        "Drawdown": drawdown
    })


        i = exit_index

    return pd.DataFrame(trades)

# The user can copy this function into their codebase to replace their current backtest_trades function.
# It now includes slippage, commission, max hold limit, position sizing, and drawdown tracking.




# === Main Pipeline ===
with st.status("Loading and processing data...", expanded=True) as status:
    raw_data = safe_download_data(interval, period)
    cleaned_data = validate_and_clean_data(raw_data)
    processed_data, features = calculate_features(cleaned_data)
    if limit_market_hours:
        processed_data = processed_data.copy()
        processed_data.index = processed_data.index.tz_convert(eastern)
        processed_data = processed_data[
            (processed_data.index.time >= market_open) & (processed_data.index.time <= market_close)
        ]
    status.update(label="Data processing complete!", state="complete")

if processed_data is None or features is None:
    st.stop()



processed_data = processed_data.copy()
processed_data.index = processed_data.index.tz_convert(eastern)
processed_data = processed_data[
    (processed_data.index.time >= market_open) & (processed_data.index.time <= market_close)
]


# ✅ Convert to local time and filter to market hours
eastern = timezone("US/Eastern")
market_open = time(9, 30)
market_close = time(16, 0)

processed_data = processed_data.copy()
processed_data.index = processed_data.index.tz_convert(eastern)
processed_data = processed_data[
    (processed_data.index.time >= market_open) & (processed_data.index.time <= market_close)
]


X, y = processed_data[features], processed_data['Target']
with st.spinner("Training model and running backtest..."):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model_dict = {
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    ),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=n_neighbors),
    "Naive Bayes": GaussianNB()
}


    model = model_dict[model_name]

    if model_name in ["Random Forest", "Gradient Boosting", "Hist Gradient Boosting"]:
        model = CalibratedClassifierCV(model, method='sigmoid', cv=3)


    model.fit(X_scaled, y)

    bt_results = backtest_trades(
    processed_data, model, scaler, features,
    return_threshold, delta_assumption, contract_cost,
    hold_mode, hold_period, confidence_threshold, confidence_mode  
)


    # Get local timezone
    local_tz = get_localzone()

    # Convert UTC time to local time
    bt_results['Time'] = bt_results['Time'].dt.tz_convert(local_tz)


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

st.subheader("💰 Backtest Review")
if bt_results.empty:
    st.write("No trades met the threshold.")
else:
    export_df = bt_results.copy()
    export_df['Time'] = export_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    tabs = st.tabs([
    "📄 Trade Log",
    "📊 Confidence Buckets",
    "📈 Equity vs SPY",
    "💹 Cumulative P&L",
    "🔍 Feature Importance",
    "📊 Model Comparison"
])



    with tabs[0]:
        st.download_button("📥 Download Full Backtest CSV", data=export_df.to_csv(index=False).encode('utf-8'),
                           file_name="backtest_results.csv", mime='text/csv')
        st.dataframe(bt_results.tail(10), use_container_width=True)
        st.metric("Total P&L", f"${bt_results['P&L'].sum():.2f}")
        st.metric("Win Rate",
        f"{(bt_results['P&L'] > 0).mean()*100:.1f}%",
        help="Percentage of trades that resulted in profit. A higher win rate suggests more consistent trade success."
    )

        st.metric("Trades", f"{len(bt_results)}")

    with tabs[1]:
        bt_results['Confidence_Bucket'] = (bt_results['Probability'] * 10).astype(int) / 10.0
        bucket_stats = bt_results.groupby('Confidence_Bucket').agg(
            Trades=('P&L', 'count'),
            Wins=('P&L', lambda x: (x > 0).sum()),
            WinRate=('P&L', lambda x: (x > 0).mean() * 100),
            AvgPnL=('P&L', 'mean')
        ).reset_index()
        st.dataframe(bucket_stats)
        fig, ax = plt.subplots()
        ax.bar(bucket_stats['Confidence_Bucket'], bucket_stats['WinRate'], width=0.05, color='skyblue')
        ax.set_xlabel('Confidence Bucket')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate by Confidence Bucket')
        ax.set_ylim(0, 100)

        # Annotate with number of trades above bars
        for i, row in bucket_stats.iterrows():
            ax.text(
                row['Confidence_Bucket'], 
                row['WinRate'] + 2, 
                f"{int(row['Trades'])} trades", 
                ha='center', 
                fontsize=8
            )

        st.pyplot(fig)


    with tabs[2]:
        equity = bt_results["P&L"].cumsum()
        equity.index = bt_results["Time"]
        price = processed_data["Close"].reindex(equity.index, method="pad")
        st.line_chart(pd.DataFrame({"Equity Curve": equity, "{selected_ticker} Price": price}))

    with tabs[3]:
        st.line_chart(bt_results["P&L"].cumsum())

    with tabs[4]:
        try:
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            top_features = [features[i] for i in sorted_idx[:10]]
            st.bar_chart(pd.Series(importances[sorted_idx[:10]], index=top_features))
        except AttributeError:
            st.warning("Model does not support feature importances (e.g., CalibratedClassifierCV wrapper).")
    with tabs[5]:
        st.subheader("📊 Model Comparison (Test Set)")

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        # Same train/test split for all models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        results = []

        for name, base_model in model_dict.items():
            try:
                # Calibrate only selected models
                if name in ["Random Forest", "Gradient Boosting", "Hist Gradient Boosting"]:
                    model_eval = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
                else:
                    model_eval = base_model

                model_eval.fit(X_train, y_train)
                y_pred = model_eval.predict(X_test)
                y_prob = model_eval.predict_proba(X_test)[:, 1]

                results.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                    "AUC": roc_auc_score(y_test, y_prob)
                })
            except Exception as e:
                st.warning(f"Model {name} failed: {str(e)}")

        results_df = pd.DataFrame(results).set_index("Model")

        st.dataframe(results_df.style.format("{:.2%}"))

        # Bar chart (AUC as default comparison)
        st.bar_chart(results_df["AUC"])



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





