import numpy as np
import pandas as pd
import os

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ret_1"]  = d["close"].pct_change(1, fill_method=None)
    d["ret_3"]  = d["close"].pct_change(3, fill_method=None)
    d["ret_12"] = d["close"].pct_change(12, fill_method=None)
    d["vol_10"] = d["ret_1"].rolling(10, min_periods=10).std()
    d["vol_30"] = d["ret_1"].rolling(30, min_periods=30).std()
    denom = (d.get("BBU", np.nan) - d.get("BBL", np.nan))
    denom = denom.replace(0, np.nan) if isinstance(denom, pd.Series) else denom
    d["%b"] = (d["close"] - d.get("BBL", pd.Series(np.nan, index=d.index))) / denom
    d["range_c"]    = (d["high"] - d["low"]) / d["close"]
    d["body"]       = (d["close"] - d["open"]) / d["open"]
    d["upper_wick"] = (d["high"] - d[["open","close"]].max(axis=1)) / d["close"]
    d["lower_wick"] = (d[["open","close"]].min(axis=1) - d["low"]) / d["close"]
    if "RSI" in d.columns:
        d["RSI_z14"] = (d["RSI"] - d["RSI"].rolling(14).mean()) / (d["RSI"].rolling(14).std())
    else:
        d["RSI_z14"] = np.nan
    return d

def get_feature_cols(results_path: str, model_name: str) -> list[str]:
    """Carga la lista de features desde el archivo generado por data_processing.py."""
    path = os.path.join(results_path, f"{model_name}_feature_cols.txt")
    alt_path = os.path.join(results_path, "feature_cols.txt")
    legacy_path = os.path.join(results_path, "XGBoost_Binario15m_feature_cols.txt")

    chosen_path = None
    if os.path.exists(path):
        chosen_path = path
    elif os.path.exists(alt_path):
        chosen_path = alt_path
    elif os.path.exists(legacy_path):
        chosen_path = legacy_path
    
    if not chosen_path:
        raise FileNotFoundError(
            f"No se encontró la lista de features en {path}, {alt_path} o {legacy_path}. "
            "Ejecuta data_processing.py primero."
        )
            
    with open(chosen_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
