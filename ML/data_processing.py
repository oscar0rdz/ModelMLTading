from __future__ import annotations
import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from ML.indicators import (
    compute_adx, compute_atr, compute_bbands, compute_macd, compute_obv, compute_rsi, _ema
)

# --- Configuración de Logging ---
logger = logging.getLogger("dataproc")
logger.setLevel(logging.INFO)
if not logger.handlers:
    hdl = logging.StreamHandler(sys.stdout)
    hdl.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(hdl)

# --- Parámetros Globales y Rutas ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "15m")
START = os.getenv("START", "2017-03-01")
END = os.getenv("END", "")
LOOK_AHEAD = int(os.getenv("LOOK_AHEAD", "3"))
K_VOL = float(os.getenv("K_VOL", "1.8"))

DATA_DIR = Path(os.getenv("DATA_DIR", "ML/data"))
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "ML/results"))
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = PROCESSED_DIR / f"{SYMBOL}_{INTERVAL}_processed.csv"
FEATURE_FILE = RESULTS_DIR / "feature_cols.txt"
FEATURE_FILE_ALIAS = RESULTS_DIR / "XGBoost_Binario15m_feature_cols.txt"

BINANCE_BASE = "https://api.binance.com"
INTERVAL_MAP = {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}


def _to_ms(s: str) -> int:
    """Convierte una fecha en string a milisegundos UTC."""
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def fetch_klines_binance(symbol: str, interval: str, start: str, end: str = "") -> pd.DataFrame:
    """
    Descarga velas OHLCV desde la API de Binance.

    Maneja la paginación de la API para obtener todos los datos en el rango
    de fechas especificado.
    """
    if interval not in INTERVAL_MAP:
        raise ValueError(f"Intervalo no soportado: {interval}")

    url = f"{BINANCE_BASE}/api/v3/klines"
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end else None
    dfs = []

    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": 1000}
        if end_ms:
            params["endTime"] = end_ms

        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        arr = resp.json()
        if not arr:
            break

        df = pd.DataFrame(arr, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        num_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]
        df[num_cols] = df[num_cols].astype(float)
        df["number_of_trades"] = df["number_of_trades"].astype(int)
        dfs.append(df[["open_time", "open", "high", "low", "close", "volume", "number_of_trades"]])

        last_open = arr[-1][0]
        if last_open == start_ms:
            break
        start_ms = last_open + 1
        time.sleep(0.1)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula y añade indicadores técnicos al DataFrame.

    Args:
        df: DataFrame con columnas OHLCV.

    Returns:
        DataFrame con los indicadores añadidos como nuevas columnas.
    """
    out = df.copy()
    out["ret_1"] = out["close"].pct_change()
    out["ret_3"] = out["close"].pct_change(3)
    out["ret_12"] = out["close"].pct_change(12)

    # Medias móviles (nombradas en mayúsculas para compatibilidad con otros módulos).
    out["SMA_10"] = out["close"].rolling(10, min_periods=10).mean()
    out["SMA_20"] = out["close"].rolling(20, min_periods=20).mean()
    out["SMA_50"] = out["close"].rolling(50, min_periods=50).mean()
    out["EMA_10"] = _ema(out["close"], 10)
    out["EMA_20"] = _ema(out["close"], 20)
    out["EMA_50"] = _ema(out["close"], 50)

    out["ATR_14"] = compute_atr(out, 14)
    out["RSI"] = compute_rsi(out, 14)

    macd, macd_sig, macd_hist = compute_macd(out, 12, 26, 9)
    out["MACD"] = macd
    out["MACDs"] = macd_sig
    out["MACDh"] = macd_hist

    mid, bbu, bbl = compute_bbands(out, 20, 2.0)
    out["BBM"] = mid
    out["BBU"] = bbu
    out["BBL"] = bbl

    out["OBV"] = compute_obv(out)
    out["ADX_14"] = compute_adx(out, 14)

    out = out.dropna().copy()
    return out


def make_labels(df: pd.DataFrame, look_ahead: int = 3, k_vol: float = 1.8) -> pd.Series:
    """
    Genera la etiqueta binaria para el modelo de clasificación.

    La etiqueta es `1` si el precio futuro supera un umbral dinámico basado
    en la volatilidad (ATR), y `0` en caso contrario.
    """
    atr = df["ATR_14"]
    close = df["close"]
    future_close = close.shift(-look_ahead)

    # El umbral es un múltiplo del ATR para adaptarse a la volatilidad del mercado.
    threshold = k_vol * (atr / close)
    target = (future_close >= close * (1 + threshold)).astype(int)
    return target


def ensure_feature_files(feature_cols: list[str]):
    """Guarda la lista de nombres de features en un archivo de texto."""
    txt = "\n".join(feature_cols)
    FEATURE_FILE.write_text(txt, encoding="utf-8")
    FEATURE_FILE_ALIAS.write_text(txt, encoding="utf-8")
    logger.info(f"Lista de features guardada en {FEATURE_FILE} y alias {FEATURE_FILE_ALIAS}")


def run(symbol: str, interval: str, start: str, end: str,
        look_ahead: int, k_vol: float, out_csv: Path):
    """
    Orquesta el proceso completo de adquisición y procesamiento de datos.
    """
    if out_csv.exists():
        logger.info(f"Usando archivo CSV existente: {out_csv}")
        df = pd.read_csv(out_csv)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    else:
        logger.info(f"Descargando {symbol} {interval} desde {start}...")
        raw = fetch_klines_binance(symbol, interval, start, end)
        if raw.empty:
            raise RuntimeError("No se obtuvieron velas. Verifica las fechas o el símbolo.")
        raw = raw.sort_values("open_time").reset_index(drop=True)
        raw.to_csv(PROCESSED_DIR / f"{symbol}_{interval}_raw.csv", index=False)
        df = raw.copy()

    df = df.set_index("open_time").sort_index()
    feats = build_features(df)
    y = make_labels(feats, look_ahead, k_vol)
    feats["target_up"] = y
    feats.reset_index().to_csv(out_csv, index=False)

    # Guarda la lista de columnas de features (excluyendo target y OHLCV).
    drop_cols = {"target_up"}
    ohlcv = {"open", "high", "low", "close", "volume", "number_of_trades"}
    feature_cols = [col for col in feats.columns if col not in drop_cols and col not in ohlcv]

    ensure_feature_files(feature_cols)
    logger.info(f"Procesamiento completado. Salida: {out_csv} ({len(feats)} filas, {len(feature_cols)} features)")


def parse_args():
    """Define y parsea los argumentos de línea de comandos."""
    ap = argparse.ArgumentParser(description="Script de procesamiento de datos para el modelo de trading.")
    ap.add_argument("--symbol", default=SYMBOL, help="Símbolo a procesar (ej. BTCUSDT).")
    ap.add_argument("--interval", default=INTERVAL, help="Intervalo de las velas (ej. 15m).")
    ap.add_argument("--start", default=START, help="Fecha de inicio (YYYY-MM-DD).")
    ap.add_argument("--end", default=END, help="Fecha de fin (opcional).")
    ap.add_argument("--lookahead", type=int, default=LOOK_AHEAD, help="Horizonte para la etiqueta.")
    ap.add_argument("--kvol", type=float, default=K_VOL, help="Multiplicador de volatilidad para la etiqueta.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
        look_ahead=args.lookahead,
        k_vol=args.kvol,
        out_csv=PROCESSED_DIR / f"{args.symbol}_{args.interval}_processed.csv"
    )
