# app.py - Streamlit-ready Signal Predictor with Buy/Sell/Hold signals

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
        macd_line = ema_fast -
