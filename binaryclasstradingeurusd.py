# app.py - Streamlit-ready Signal Predictor

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings("ignore")

st.title("Week 2 Signal Predictor")

# -------------------------
# Dataset upload
# -------------------------
uploaded_file = st.file_uploader("Upload Week 1 dataset CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=True)
    st.write("Dataset loaded. Shape:", df.shape)
    st.dataframe(df.head())

    # -------------------------
    # Feature Engineering
    # -------------------------
    LOOKAHEAD = 10
    UP_TH = 0.01
    DOWN_TH = -0.01

    # Technical indicators
    def rsi(series, window=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def macd(series, span_fast=12, span_slow=26, span_signal=9):
        ema_fast = series.ewm(span=span_fast, adjust=False).mean()
        ema_slow = series.ewm(span=span_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=span_signal, adjust=False).mean()
        return macd_line, signal_line

    def atr(high, low, close, window=14):
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    def stochastic_k(high, low, close, k_window=14):
        lowest_low = low.rolling(k_window).min()
        highest_high = high.rolling(k_window).max()
        return 100 * (close - lowest_low) / (highest_high - lowest_low)

    # Price-based features
    df['ret'] = df['Close'].pct_change()
    df['logret'] = np.log(df['Close']).diff()

    # Moving averages
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()

    # Technical indicators
    df['rsi_14'] = rsi(df['Close'], 14)
    df['macd'], df['macd_signal'] = macd(df['Close'])
    df['stoch_k'] = stochastic_k(df['High'], df['Low'], df['Close'], 14)
    df['atr_14'] = atr(df['High'], df['Low'], df['Close'], 14)

    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df[f'ret_lag_{lag}'] = df['ret'].shift(lag)

    # Structural features from Week-1
    possible_struct = ['Swing_High', 'Swing_Low', 'Support_Level', 'Resistance_Level',
                       'Supply_Zone_Width', 'Demand_Zone_Width', 'Nearest_Fib_Dist']
    struct_cols = [c for c in possible_struct if c in df.columns]

    st.write("Using structural columns:", struct_cols)

    # Label creation
    df['fwd_close'] = df['Close'].shift(-LOOKAHEAD)
    df['fwd_ret'] = (df['fwd_close'] - df['Close']) / df['Close']

    def label_from_ret(x):
        if x >= UP_TH:
            return 1      # Buy
        elif x <= DOWN_TH:
            return -1     # Sell
        else:
            return 0     # Hold

    df['label'] = df['fwd_ret'].apply(lambda x: label_from_ret(x) if pd.notnull(x) else np.nan)

    feature_cols = ['ret','logret','sma_10','sma_50','ema_12','rsi_14',
                    'macd','macd_signal','stoch_k','atr_14'] + \
                   [f'ret_lag_{lag}' for lag in [1,2,3,5,10]] + struct_cols

    df_model = df[feature_cols + ['label']].dropna()
    st.write("Label distribution:")
    st.bar_chart(df_model['label'].value_counts())

    # -------------------------
    # Prepare data for model
    # -------------------------
    X = df_model.drop(['label'], axis=1)
    y = df_model['label']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # Drop missing values
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]
    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]

    # -------------------------
    # Train model
    # -------------------------
    model = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # -------------------------
    # Display evaluation
    # -------------------------
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(model, 'lgbm_model.pkl')
    st.write("Model saved as lgbm_model.pkl")

    # Feature importance plot
    st.subheader("Top 20 Feature Importances")
    fig, ax = plt.subplots(figsize=(10,6))
    lgb.plot_importance(model, max_num_features=20, importance_type='gain', ax=ax)
    st.pyplot(fig)

