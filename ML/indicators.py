# ML/indicators.py
"""
Módulo centralizado para el cálculo de indicadores técnicos.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd

# --------------------------- Indicators -------------------------------
def _ema(x: pd.Series, span: int) -> pd.Series:
    """Calcula la Media Móvil Exponencial (EMA)."""
    return x.ewm(span=span, adjust=False).mean()

def compute_atr(df: pd.DataFrame, n: int=14) -> pd.Series:
    """Calcula el Average True Range (ATR)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=n).mean()
    return atr

def compute_rsi(df: pd.DataFrame, n: int=14) -> pd.Series:
    """Calcula el Relative Strength Index (RSI)."""
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(df: pd.DataFrame, fast: int=12, slow: int=26, signal: int=9) -> Tuple[pd.Series,pd.Series,pd.Series]:
    """Calcula Moving Average Convergence Divergence (MACD)."""
    close = df["close"]
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def compute_adx(df: pd.DataFrame, n: int=14) -> pd.Series:
    """Calcula el Average Directional Index (ADX)."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move>down_move) & (up_move>0), up_move, 0.0)
    minus_dm = np.where((down_move>up_move) & (down_move>0), down_move, 0.0)
    tr = pd.concat([(high-low).abs(), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=n).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(n, min_periods=n).sum() / atr.replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(n, min_periods=n).sum() / atr.replace(0, np.nan))
    dx = 100 * ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) )
    adx = dx.rolling(n, min_periods=n).mean()
    return adx

def compute_obv(df: pd.DataFrame) -> pd.Series:
    """Calcula el On-Balance Volume (OBV)."""
    close = df["close"]
    volume = df["volume"]
    delta = close.diff().fillna(0.0)
    direction = np.sign(delta)
    obv = (direction * volume).fillna(0.0).cumsum()
    return obv

def compute_bbands(df: pd.DataFrame, n: int=20, num_std: float=2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calcula las Bandas de Bollinger (BBands)."""
    close = df["close"]
    mid = close.rolling(n, min_periods=n).mean()
    std = close.rolling(n, min_periods=n).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower
