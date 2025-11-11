#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Walk-forward analysis (WFA) para datos de 15m con una etiqueta de 3 velas.
El script entrena un clasificador XGBoost y aplica compuertas cuantiles/EV para controlar la densidad de señales.
"""

from __future__ import annotations
import os
import sys
import json
import math
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt





logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("wfa_3bar_xgb")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XGB_PARAM_FILE = os.path.join(BASE_DIR, "results", "btc15m_xgb_calibrated_h3_model_params.txt")
XGB_PARAM_FILE = os.path.expanduser(os.getenv("WFA_XGB_PARAM_FILE", DEFAULT_XGB_PARAM_FILE))


def _parse_param_value(raw: str):
    """Interpreta valores de texto (números/bools) provenientes de archivos de configuración."""
    val = raw.strip()
    if not val:
        return val
    low = val.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if any(c in val for c in (".", "e", "E")):
            fval = float(val)
            return int(fval) if fval.is_integer() else fval
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def _load_external_xgb_params(path: str) -> Dict[str, object]:
    """Lee hiperparámetros afinados desde un archivo plano clave:valor."""
    if not path or not os.path.exists(path):
        return {}
    params: Dict[str, object] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                params[key.strip()] = _parse_param_value(val)
        logger.info("Hiperparámetros XGB externos cargados desde %s", path)
    except Exception as exc:
        logger.warning("No se pudieron cargar hiperparámetros desde %s: %s", path, exc)
    return params


TUNED_XGB_PARAMS = _load_external_xgb_params(XGB_PARAM_FILE)
_XGB_OVERRIDES_ENV = os.getenv("WFA_XGB_PARAM_OVERRIDES")
if _XGB_OVERRIDES_ENV:
    try:
        override_dict = json.loads(_XGB_OVERRIDES_ENV)
        if isinstance(override_dict, dict):
            TUNED_XGB_PARAMS.update(override_dict)
            logger.info("Overrides adicionales aplicados a los hiperparámetros XGB (WFA_XGB_PARAM_OVERRIDES).")
    except json.JSONDecodeError as exc:
        logger.warning("No se pudo parsear WFA_XGB_PARAM_OVERRIDES como JSON: %s", exc)


# Ventanas WFA (pensadas para 15m)
TRAIN_BARS = 9000   # ~ 92 días de train
TEST_BARS  = 2500   # ~ 10 días de test
STEP_BARS  = 500   # avanzar 5 días por iteración

# Trading params
INITIAL_CAPITAL   = 1000.0
COMMISSION_RATE   = 0.0002     # 2 bps por lado (≈0.02%)
SLIPPAGE_PCT      = 0.0001     # 1 bps

# Estrategia 3 velas (horizonte y gestión)
HORIZON_BARS      = 3
TP_BPS            = 26        # 0.26%
SL_BPS            = 10       # 0.10%
MAX_BARS_IN_TRADE = HORIZON_BARS

# Gestión defensiva
EARLY_TP_BPS       = 3.0        # toma de ganancia temprana si toca en la 1ª barra
EARLY_TP_BARS      = 1          # sólo válida en la primer barra tras la entrada
ONE_BAR_FAIL_FRAC  = 0.70       # si baja 80% del rango entry->SL en 1 barra, salir
BE_TRIGGER_FRAC    = 0.30       # a +30% del camino a TP, mover a break-even
TSL_ACTIVATION_MULT= 1.00       # activa TSL tras BE (proporción del camino a TP)
TSL_TRAIL_MULT     = 0.50       # trailing basado en (TP-Entry)

def _cost_bps_from_env() -> float:
    """Calcula el costo efectivo en bps (maker ida/vuelta + slippage + estrés + spread)."""
    maker_bps = float(os.getenv("MAKER_FEE_BPS", "1.0"))
    slipp_bps = float(os.getenv("SLIPPAGE_BPS", "1.0"))
    stress_bps = float(os.getenv("STRESS_COST_BPS", "1.0"))
    spread_bps = float(os.getenv("SPREAD_MIN_BPS", "0.0"))
    return maker_bps * 2.0 + slipp_bps + stress_bps + spread_bps


# Gating por EV (esperanza en bps)
EV_ENABLED = True
_COST_DEFAULT_BPS = _cost_bps_from_env()
try:
    COST_BPS = float(os.getenv("BACKTEST_COST_BPS", "").strip() or _COST_DEFAULT_BPS)
except ValueError:
    COST_BPS = _COST_DEFAULT_BPS
_EV_MARGIN_BASE = max(3.5, COST_BPS * 0.75)
EV_MARGIN_BPS = float(os.getenv("EV_MARGIN_BPS", str(_EV_MARGIN_BASE)))

# Compuerta horaria
HOUR_GATE_MIN_SIGNALS = int(os.getenv("HOUR_GATE_MIN_SIGNALS", "24"))
HOUR_GATE_MIN_COVERAGE = float(os.getenv("HOUR_GATE_MIN_COVERAGE", "0.30"))

# Position sizing (risk-based)
RISK_PER_TRADE      = 0.008   # % del capital
MIN_RISK_PER_TRADE  = 0.0015

# Señal por cuantiles (garantizar actividad)
# Selección de umbral
SIGNAL_POLICY   = "ev_search"    # ["quantile", "fixed", "ev_search"]
SIGNAL_Q        = 0.86          # percentil base si se usa cuantiles
FIXED_THRESHOLD = 0.68      # si SIGNAL_POLICY=="fixed"
THRESHOLD_GRID_QS = [
    0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90
]
THRESHOLD_MIN_SIGNALS = 60       # mínimo absoluto de señales en train para aceptar el umbral
THRESHOLD_MIN_SIGNAL_FRAC = 0.015  # % mínimo del dataset (ej. 1.5% de train) para evitar muestras diminutas
THRESHOLD_MIN_EV_BPS  = 1.0       # EV mínimo (train) para operar el bloque

# Filtros de tendencia
SHORT_MA = 20
LONG_MA  = 50

# Salida
OUTDIR = "results_wfa"

# Probabilidad/calibración
USE_WFA_PROB_CALIB = False

# Control de riesgo agregado
CAPITAL_STOP_MULTIPLIER = float(os.getenv("CAPITAL_STOP_MULT", "0.50"))  # detener si capital < 50% inicial

WFA_CALIB_FRAC = float(os.getenv("WFA_CALIB_FRAC", "0.25"))
WFA_CALIB_MIN = int(os.getenv("WFA_CALIB_MIN", "400"))
WFA_THRESHOLD_MIN_SAMPLES = int(os.getenv("WFA_THRESHOLD_MIN_SAMPLES", "800"))
BLOCK_DD_STOP_PCT = float(os.getenv("BLOCK_DD_STOP_PCT", "0.15"))
PROB_MIN = float(os.getenv("PROB_MIN", "0.55"))
_prob_max_raw = os.getenv("PROB_MAX")
if _prob_max_raw is not None:
    _prob_max_raw = _prob_max_raw.strip()
PROB_MAX = float(_prob_max_raw) if _prob_max_raw else None
HOLDOUT_EV_MIN_BPS = float(os.getenv("HOLDOUT_EV_MIN_BPS", "10.0"))
HOLDOUT_MIN_TRADES = int(os.getenv("HOLDOUT_MIN_TRADES", "25"))
HOLDOUT_QS = np.array([
    0.98, 0.975, 0.95, 0.93, 0.90, 0.88, 0.85, 0.82, 0.80, 0.78, 0.75,
    0.72, 0.70, 0.65, 0.60
], dtype=float)
MAX_TRADES_PER_BLOCK = int(os.getenv("MAX_TRADES_PER_BLOCK", "60"))


def _bps_to_mult(bps: float) -> float:
    """Convierte bps a un multiplicador (>1 implica alza)."""
    return 1.0 + bps / 10000.0


def expected_ev_bps(p: np.ndarray, tp_bps: float, sl_bps: float, cost_bps: float) -> np.ndarray:
    """Calcula la EV esperada en bps: p*TP - (1-p)*SL - costos."""
    p = np.asarray(p)
    return p * float(tp_bps) - (1.0 - p) * float(sl_bps) - float(cost_bps)


def choose_threshold_by_ev(probs_train: np.ndarray, tp_bps: float, sl_bps: float, cost_bps: float) -> float:
    """Selecciona un umbral mediante relajación cuantílica maximizando EV en TRAIN.
    Implementa el selector ALFA-02: se reduce el cuantil hasta cumplir densidad y EV mínima.
    """
    q_init = float(os.getenv("EV_Q", "0.82"))
    q_step = float(os.getenv("EV_RELAX_Q_STEP", "0.02"))
    q_floor = float(os.getenv("EV_RELAX_Q_FLOOR", "0.78"))
    ev_need = float(os.getenv("EV_MARGIN_BPS", str(EV_MARGIN_BPS)))
    min_n = int(os.getenv("EV_MIN_TRADES_PER_BLOCK", "15"))
    prob_min = float(os.getenv("PROB_MIN", "0.55"))

    probs = np.asarray(probs_train)
    q = q_init
    thr_best = None
    ev_best = -1e12
    # iterate downwards from q_init to q_floor
    while q + 1e-12 >= q_floor:
        thr = float(np.quantile(probs, q))
        thr = max(min(thr, 0.95), prob_min)
        mask = probs >= thr
        n = int(mask.sum())
        if n > 0:
            evs = expected_ev_bps(probs[mask], tp_bps, sl_bps, cost_bps)
            ev_mean = float(np.mean(evs))
        else:
            ev_mean = -1e12

        ok = (ev_mean >= ev_need) and (n >= min_n)
        if ok and (ev_mean > ev_best):
            ev_best = ev_mean
            thr_best = thr

        q -= q_step

    if thr_best is None:
        thr_best = max(min(float(np.quantile(probs, q_floor)), 0.95), prob_min)
    return float(thr_best)


def bps_to_mult(bps: float) -> float:
    """Convierte bps a multiplicador de precio (25 bps => 1.0025)."""
    return 1.0 + (bps / 10000.0)


def relax_threshold_if_dry(probs_test: np.ndarray, thr_base: float) -> float:
    """Relaja el umbral hacia abajo si en TEST hay pocas o ninguna señal.
    Devuelve un umbral >= prob_min donde se alcanzan al menos MIN_TRADES_TEST señales
    o el valor mínimo prob_min si no es posible.
    """
    probs = np.asarray(probs_test)
    min_trades = int(os.getenv("MIN_TRADES_TEST", "8"))
    floor = float(os.getenv("PROB_MIN", "0.55"))
    step = float(os.getenv("RELAX_STEP", "0.02"))

    thr = float(thr_base)
    def count(t):
        return int((probs >= t).sum())

    if count(thr) >= min_trades:
        return thr

    t = thr
    while t > floor and count(t) < min_trades:
        t -= step
    t = max(t, floor)
    return float(t)


def clamp_prob_threshold(thr: float) -> float:
    """Aplica límites globales a un umbral de probabilidad."""
    thr = max(PROB_MIN, float(thr))
    if PROB_MAX is not None:
        thr = min(thr, PROB_MAX)
    return thr


def _ensure_bool_series(mask, index: pd.Index) -> pd.Series:
    """Normaliza cualquier máscara a una Serie booleana alineada al índice dado."""
    if isinstance(mask, pd.Series):
        return mask.reindex(index, fill_value=False).astype(bool)
    if isinstance(mask, np.ndarray):
        if mask.dtype != bool:
            mask = mask.astype(bool)
        return pd.Series(mask, index=index)
    return pd.Series(bool(mask), index=index)


def _evaluate_signal_flow(seg: pd.DataFrame, thr: float, gate_mask: pd.Series, trend_mask: pd.Series, hour_mask: pd.Series) -> Tuple[pd.Series, dict]:
    """Aplica cada filtro secuencialmente y devuelve el flujo de conteos."""
    thr_mask = (seg["prob"] >= thr)
    ev_mask = thr_mask & gate_mask
    trend_applied = ev_mask & trend_mask
    final_mask = trend_applied & hour_mask
    diag = {
        "threshold": int(thr_mask.sum()),
        "after_ev": int(ev_mask.sum()),
        "after_trend": int(trend_applied.sum()),
        "after_hours": int(final_mask.sum()),
    }
    return final_mask, diag


def build_signals_with_relaxations(
    seg: pd.DataFrame,
    base_thr: float,
    trend_mask,
    gate_mask,
    hour_mask,
    min_trades: int,
    ev_gate_enabled: bool = True,
) -> Tuple[pd.Series, float, dict, dict, dict, List[str]]:
    """Construye la máscara final de señales aplicando relajaciones ordenadas.
    Enlaza compuertas de umbral, EV, tendencia y horario hasta lograr el mínimo de operaciones requerido.
    Devuelve: máscara final, umbral final, diagnósticos base/final, estado y lista de relajaciones.
    """
    idx = seg.index
    thr = float(base_thr)
    gate_series = _ensure_bool_series(gate_mask, idx)
    trend_series = _ensure_bool_series(trend_mask, idx)
    hour_series = _ensure_bool_series(hour_mask, idx)
    min_trades = max(int(min_trades), 0)

    relaxations: List[str] = []
    trend_mode = "strict"
    thr_relaxed = False

    mask, diag = _evaluate_signal_flow(seg, thr, gate_series, trend_series, hour_series)
    base_diag = dict(diag)

    def need_more() -> bool:
        return diag["after_hours"] < min_trades

    if need_more() and not hour_series.all():
        hour_series = pd.Series(True, index=idx)
        relaxations.append("hour_gate_off")
        mask, diag = _evaluate_signal_flow(seg, thr, gate_series, trend_series, hour_series)

    if need_more():
        tol = float(os.getenv("TREND_RELAX_TOLERANCE", "0.01"))
        relaxed_trend = seg[f"sma{SHORT_MA}"] >= (seg[f"sma{LONG_MA}"] * (1.0 - tol))
        if relaxed_trend.any():
            trend_series = _ensure_bool_series(relaxed_trend, idx)
            trend_mode = "relaxed"
            relaxations.append(f"trend_relaxed_tol={tol:.4f}")
            mask, diag = _evaluate_signal_flow(seg, thr, gate_series, trend_series, hour_series)

    if need_more() and not trend_series.all():
        trend_series = pd.Series(True, index=idx)
        trend_mode = "off"
        relaxations.append("trend_filter_off")
        mask, diag = _evaluate_signal_flow(seg, thr, gate_series, trend_series, hour_series)

    if need_more():
        thr_new = relax_threshold_if_dry(seg["prob"].values, thr)
        if thr_new < thr - 1e-9:
            thr = float(thr_new)
            thr_relaxed = True
            relaxations.append(f"threshold_relaxed={thr:.4f}")
            mask, diag = _evaluate_signal_flow(seg, thr, gate_series, trend_series, hour_series)

    if need_more() and ev_gate_enabled and not gate_series.all():
        gate_series = pd.Series(True, index=idx)
        relaxations.append("ev_gate_off")
        mask, diag = _evaluate_signal_flow(seg, thr, gate_series, trend_series, hour_series)

    state = {
        "trend_mode": trend_mode,
        "hour_gate_active": bool((~hour_series).any()),
        "ev_gate_active": ev_gate_enabled and bool((~gate_series).any()),
        "threshold_relaxed": thr_relaxed,
        "min_trades": min_trades,
        "final_signals": diag["after_hours"],
    }
    return mask, thr, diag, base_diag, state, relaxations


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina valores infinitos o NaN para evitar fallos aguas abajo."""
    df = df.replace([np.inf, -np.inf], np.nan).copy()
    df = df.dropna()
    return df


def sma(s: pd.Series, n: int) -> pd.Series:
    """Media móvil simple sobre n barras."""
    return s.rolling(n).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Índice RSI clásico basado en medias móviles."""
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_down = pd.Series(down, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR suavizado para medir rango verdadero."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores técnicos base que alimentan al modelo."""
    df = df.copy()
    df[f"sma{SHORT_MA}"] = sma(df["close"], SHORT_MA)
    df[f"sma{LONG_MA}"]  = sma(df["close"], LONG_MA)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, 14)

    # Retornos/momentum simples
    df["ret1"] = df["close"].pct_change(1)
    df["ret3"] = df["close"].pct_change(3)
    df["ret6"] = df["close"].pct_change(6)
    df["vol_chg"] = df["volume"].pct_change(3)

    return df



def label_tp_sl_3bar(df: pd.DataFrame) -> pd.DataFrame:
    """Genera la etiqueta binaria para la estrategia path-aware de 3 velas (solo largos).
    - label=1 si el TP se toca dentro de HORIZON_BARS sin tocar el SL.
    - label=0 si el SL se toca dentro de HORIZON_BARS sin tocar el TP.
    - Casos ambiguos (se tocan ambos niveles) se marcan como NaN para omitirlos.
    """
    df = df.copy()
    close = df["close"]

    tp_mult = bps_to_mult(TP_BPS)
    sl_mult = 1.0 / bps_to_mult(SL_BPS)  # equivalente a -SL_BPS

    # Ventanas futuras
    fut_high = df["high"].rolling(window=HORIZON_BARS, min_periods=HORIZON_BARS).max().shift(-HORIZON_BARS+1)
    fut_low  = df["low"].rolling(window=HORIZON_BARS, min_periods=HORIZON_BARS).min().shift(-HORIZON_BARS+1)

    tp_price = close * tp_mult
    sl_price = close * sl_mult

    hit_tp = fut_high >= tp_price
    hit_sl = fut_low  <= sl_price

    label = pd.Series(np.nan, index=df.index)
    label[(hit_tp) & (~hit_sl)] = 1
    label[(hit_sl) & (~hit_tp)] = 0

    df["label"] = label
    return df



FEATURES = [
    "open", "high", "low", "close", "volume",
    f"sma{SHORT_MA}", f"sma{LONG_MA}", "rsi14", "atr14",
    "ret1", "ret3", "ret6", "vol_chg",
]


def prep_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepara X/y filtrando características y etiquetas válidas para el modelo."""
    X = df[FEATURES].copy()
    y = df["label"].copy()
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    X = handle_missing(X)
    y = y.loc[X.index]
    return X, y


def build_model() -> Pipeline:
    """Construye el pipeline de preprocesado + XGBoost."""
    pre = ColumnTransformer([
        ("num", RobustScaler(), FEATURES),
    ], remainder="drop")

    # Parámetros base; se pueden sobrescribir vía archivo o env.
    base_params: Dict[str, object] = {
        "objective": "binary:logistic",
        "n_estimators": 320,
        "learning_rate": 0.02,
        "max_depth": 5,
        "subsample": 0.75,
        "colsample_bytree": 0.78,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "min_child_weight": 6,
        "gamma": 0.0,
        "max_delta_step": 0,
        "random_state": 42,
        "eval_metric": "logloss",
        "n_jobs": 4,
        "tree_method": os.getenv("XGB_TREE_METHOD", "hist"),
    }
    if TUNED_XGB_PARAMS:
        base_params.update(TUNED_XGB_PARAMS)

    xgb = XGBClassifier(**base_params)

    return Pipeline([("pre", pre), ("xgb", xgb)])



def position_size(capital: float, entry: float, stop: float, consecutive_losses: int = 0) -> float:
    """Calcula el tamaño de posición respetando el riesgo por operación."""
    reduce = (0.9 ** consecutive_losses)
    capital_ratio = min(1.0, max(0.1, capital / INITIAL_CAPITAL))
    risk_eff = max(RISK_PER_TRADE * reduce * capital_ratio, MIN_RISK_PER_TRADE)
    risk_amt = capital * risk_eff
    dist = max(entry - stop, 1e-8)
    size = risk_amt / dist
    # cap notional 30% capital
    max_notional = 0.30 * capital
    return min(size, max_notional / entry)



@dataclass
class Trade:
    """Representa una operación individual (entrada, gestión y salida)."""
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    commission_entry: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    commission_exit: Optional[float] = None
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None
    realized_usd: float = 0.0        # PnL parcial realizado
    be_active: bool = False         # break-even activado
    model_prob: float = 0.0
    model_ev_bps: float = 0.0
    signal_note: str = ""

@dataclass
class ThresholdInfo:
    """Agrupa la información del selector de umbral y su guardia holdout."""
    threshold: float
    win_rate: float
    ev_bps: float
    signals: int
    policy: str
    used_fallback: bool = False
    skip_block: bool = False
    holdout_win_rate: float = float("nan")
    holdout_ev_bps: float = float("nan")
    holdout_signals: int = 0
    holdout_note: str = ""


def _quantile_threshold(probs: np.ndarray, labels: np.ndarray, quantile: float, policy_name: str, used_fallback: bool) -> ThresholdInfo:
    """Evalúa un umbral cuantílico básico y devuelve sus métricas."""
    q = min(max(quantile, 0.0), 1.0)
    thr = float(np.quantile(probs, q))
    mask = probs >= thr
    signals = int(mask.sum())
    if signals == 0:
        win_rate = 0.0
    else:
        win_rate = float(np.nanmean(labels[mask]))
    cost_bps = _cost_bps_from_env()
    ev_bps = win_rate * TP_BPS - (1.0 - win_rate) * SL_BPS - cost_bps
    skip_block = ev_bps < THRESHOLD_MIN_EV_BPS
    return ThresholdInfo(
        threshold=thr,
        win_rate=win_rate,
        ev_bps=ev_bps,
        signals=signals,
        policy=policy_name,
        used_fallback=used_fallback,
        skip_block=skip_block,
    )


def _ev_search_threshold(probs: np.ndarray, labels: np.ndarray) -> Optional[ThresholdInfo]:
    """Busca el mejor umbral explorando una grilla y maximizando EV + densidad."""
    qs = np.clip(np.asarray(THRESHOLD_GRID_QS, dtype=float), 0.0, 1.0)
    candidates = np.unique(np.quantile(probs, qs))
    n_labels = max(len(labels), 1)
    min_signals_required = max(
        THRESHOLD_MIN_SIGNALS,
        int(np.ceil(n_labels * THRESHOLD_MIN_SIGNAL_FRAC)),
    )

    best: Optional[ThresholdInfo] = None
    for thr in np.sort(candidates)[::-1]:
        mask = probs >= thr
        signals = int(mask.sum())
        if signals < min_signals_required:
            continue
        win_rate = float(np.nanmean(labels[mask]))
        if math.isnan(win_rate):
            continue
        cost_bps = _cost_bps_from_env()
        ev_bps = win_rate * TP_BPS - (1.0 - win_rate) * SL_BPS - cost_bps
        info = ThresholdInfo(
            threshold=float(thr),
            win_rate=win_rate,
            ev_bps=ev_bps,
            signals=signals,
            policy="ev_search",
            used_fallback=False,
            skip_block=ev_bps < THRESHOLD_MIN_EV_BPS,
        )
        if best is None or ev_bps > best.ev_bps or (math.isclose(ev_bps, best.ev_bps) and thr > best.threshold):
            best = info
    return best


def _apply_holdout_guard(info: ThresholdInfo, holdout_probs: Optional[np.ndarray], holdout_labels: Optional[np.ndarray]) -> ThresholdInfo:
    """Revalida el umbral con un holdout; desactiva bloques débiles si es necesario."""
    if holdout_probs is None or holdout_labels is None:
        info.holdout_note = "holdout_missing"
        return info

    hp = np.asarray(holdout_probs, dtype=float)
    hl = np.asarray(holdout_labels, dtype=float)
    mask = np.isfinite(hp) & np.isfinite(hl)
    hp = hp[mask]
    hl = hl[mask]
    if len(hp) < HOLDOUT_MIN_TRADES:
        info.holdout_note = f"holdout_small({len(hp)})"
        return info

    cost_bps = _cost_bps_from_env()
    candidate_thrs = np.unique(np.concatenate((
        np.array([info.threshold], dtype=float),
        np.quantile(hp, HOLDOUT_QS)
    )))
    candidate_thrs = [clamp_prob_threshold(t) for t in candidate_thrs]
    candidate_thrs = sorted(set(candidate_thrs), reverse=True)

    last_stats = None
    for thr in candidate_thrs:
        mask_thr = hp >= thr
        signals = int(mask_thr.sum())
        if signals < HOLDOUT_MIN_TRADES:
            continue
        win_rate = float(np.nanmean(hl[mask_thr])) if signals else 0.0
        ev_bps = win_rate * TP_BPS - (1.0 - win_rate) * SL_BPS - cost_bps
        last_stats = (thr, signals, win_rate, ev_bps)
        if ev_bps >= HOLDOUT_EV_MIN_BPS:
            info.threshold = float(thr)
            info.holdout_signals = signals
            info.holdout_win_rate = win_rate
            info.holdout_ev_bps = ev_bps
            info.holdout_note = "holdout_pass"
            info.skip_block = False
            return info

    if last_stats is None:
        info.holdout_note = "holdout_no_candidates"
        info.skip_block = True
        info.holdout_signals = 0
        info.holdout_ev_bps = float("nan")
        info.holdout_win_rate = float("nan")
        return info

    thr, signals, win_rate, ev_bps = last_stats
    info.threshold = float(thr)
    info.holdout_signals = signals
    info.holdout_win_rate = win_rate
    info.holdout_ev_bps = ev_bps
    info.holdout_note = "holdout_fail"
    info.skip_block = True
    return info


def compute_threshold_from_train(
    probs_train: np.ndarray,
    labels_train: np.ndarray,
    holdout_probs: Optional[np.ndarray] = None,
    holdout_labels: Optional[np.ndarray] = None,
) -> ThresholdInfo:
    """Determina el umbral óptimo con datos de train y aplica la guardia de holdout."""
    probs = np.asarray(probs_train, dtype=float)
    labels = np.asarray(labels_train, dtype=float)
    mask = np.isfinite(probs) & np.isfinite(labels)
    probs = probs[mask]
    labels = labels[mask]
    if len(probs) == 0:
        info = ThresholdInfo(
            threshold=1.0,
            win_rate=0.0,
            ev_bps=-float("inf"),
            signals=0,
            policy="none",
            used_fallback=True,
            skip_block=True,
        )
        return _apply_holdout_guard(info, holdout_probs, holdout_labels)

    if SIGNAL_POLICY == "fixed":
        thr = float(FIXED_THRESHOLD)
        mask = probs >= thr
        signals = int(mask.sum())
        win_rate = float(np.nanmean(labels[mask])) if signals else 0.0
        ev_bps = win_rate * TP_BPS - (1.0 - win_rate) * SL_BPS - COST_BPS
        info = ThresholdInfo(
            threshold=thr,
            win_rate=win_rate,
            ev_bps=ev_bps,
            signals=signals,
            policy="fixed",
            used_fallback=False,
            skip_block=ev_bps < THRESHOLD_MIN_EV_BPS,
        )
        return _apply_holdout_guard(info, holdout_probs, holdout_labels)

    if SIGNAL_POLICY == "ev_search":
        info = _ev_search_threshold(probs, labels)
        if info is not None:
            return _apply_holdout_guard(info, holdout_probs, holdout_labels)
        # fallback a cuantiles si no hay suficientes señales
        return _apply_holdout_guard(
            _quantile_threshold(probs, labels, SIGNAL_Q, "ev_fallback", True),
            holdout_probs,
            holdout_labels,
        )

    # política por cuantiles
    # Si la política EV_POLICY está activada a 'quantile', usamos el selector EV-aware
    if os.getenv("EV_POLICY", "quantile") == "quantile":
        try:
            cost_bps = _cost_bps_from_env()
            thr = choose_threshold_by_ev(probs, TP_BPS, SL_BPS, cost_bps)
            mask = probs >= thr
            signals = int(mask.sum())
            win_rate = float(np.nanmean(labels[mask])) if signals else 0.0
            ev_bps = win_rate * TP_BPS - (1.0 - win_rate) * SL_BPS - cost_bps
            info = ThresholdInfo(
                threshold=float(thr),
                win_rate=win_rate,
                ev_bps=ev_bps,
                signals=signals,
                policy="ev_quantile",
                used_fallback=False,
                skip_block=ev_bps < THRESHOLD_MIN_EV_BPS,
            )
            return _apply_holdout_guard(info, holdout_probs, holdout_labels)
        except Exception:
            pass

    return _apply_holdout_guard(
        _quantile_threshold(probs, labels, SIGNAL_Q, "quantile", False),
        holdout_probs,
        holdout_labels,
    )


def backtest_segment(df: pd.DataFrame, model: Pipeline, train_df: pd.DataFrame, initial_capital: float) -> Tuple[pd.DataFrame, pd.DataFrame, float, ThresholdInfo]:
    """Ejecuta el backtest en un bloque test usando el modelo y umbral derivados del bloque train."""
    seg = df.copy()

    # Probabilidades de train (para umbral por cuantiles)
    Xtr, ytr = prep_xy(train_df)
    n_train = len(Xtr)
    thr_min_samples = max(100, WFA_THRESHOLD_MIN_SAMPLES)
    calib_min = max(0, WFA_CALIB_MIN)

    calib_len = 0
    if n_train > thr_min_samples and calib_min > 0:
        calib_len = max(int(round(n_train * WFA_CALIB_FRAC)), calib_min)
        calib_len = min(calib_len, max(0, n_train - thr_min_samples))
        if calib_len < calib_min:
            calib_len = 0
    split_idx = max(0, n_train - calib_len)
    if split_idx == 0:
        split_idx = n_train
        calib_len = 0

    X_thr = Xtr.iloc[:split_idx]
    y_thr = ytr.iloc[:split_idx]
    X_cal = Xtr.iloc[split_idx:] if calib_len > 0 else None
    y_cal = ytr.iloc[split_idx:] if calib_len > 0 else None

    p_train_raw_all = model.predict_proba(Xtr)[:, 1]
    p_train = p_train_raw_all.copy()
    calibrator: Optional[IsotonicRegression] = None

    can_calibrate = (
        USE_WFA_PROB_CALIB
        and X_cal is not None
        and len(X_cal) >= calib_min
        and len(np.unique(y_cal)) > 1
    )
    if can_calibrate:
        try:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(p_train_raw_all[split_idx:], y_cal.to_numpy())
            p_train = calibrator.predict(p_train_raw_all)
            holdout_pct = (len(X_cal) / max(1, n_train)) * 100.0
            logger.info(
                "Calibración isotónica aplicada en TRAIN (WFA) con %.1f%% del bloque como holdout.",
                holdout_pct,
            )
        except Exception as exc:
            calibrator = None
            logger.warning("No se pudo calibrar probabilidades en TRAIN (WFA): %s", exc)
    elif USE_WFA_PROB_CALIB:
        reason = "clases degeneradas" if len(np.unique(ytr)) <= 1 else "muestras holdout insuficientes"
        logger.info("Calibración isotónica omitida en TRAIN (WFA): %s.", reason)

    p_thr = p_train[:len(X_thr)]
    holdout_probs = p_train[len(X_thr):] if len(X_thr) < len(p_train) else None
    holdout_labels = ytr.iloc[len(X_thr):] if len(X_thr) < len(ytr) else None
    holdout_labels_arr = holdout_labels.to_numpy() if holdout_labels is not None and len(holdout_labels) else None

    thr_info = compute_threshold_from_train(
        p_thr,
        y_thr.to_numpy(),
        holdout_probs,
        holdout_labels_arr,
    )

    # Probabilidades de test
    Xte = seg[FEATURES].copy()
    Xte = handle_missing(Xte)
    p_test_raw = model.predict_proba(Xte)[:, 1]
    if calibrator is not None:
        try:
            p_test = calibrator.predict(p_test_raw)
        except Exception as exc:
            logger.warning("No se pudo aplicar calibración en TEST (WFA): %s", exc)
            p_test = p_test_raw
    else:
        p_test = p_test_raw

    # Filtro tendencia
    trend = seg[f"sma{SHORT_MA}"] > seg[f"sma{LONG_MA}"]

    # Compute threshold from train and clamp to reasonable limits
    thr = clamp_prob_threshold(float(thr_info.threshold))
    # keep a copy of the clamped threshold before any relaxations are applied
    thr_clamped = float(thr)
    holdout_win_pct = thr_info.holdout_win_rate * 100.0 if np.isfinite(thr_info.holdout_win_rate) else float("nan")
    holdout_ev_bps = thr_info.holdout_ev_bps if np.isfinite(thr_info.holdout_ev_bps) else float("nan")
    logger.info(
        "Guardia holdout: note=%s | señales=%d | win=%.2f%% | ev=%.2fbps | thr=%.4f",
        thr_info.holdout_note,
        thr_info.holdout_signals,
        holdout_win_pct,
        holdout_ev_bps,
        thr,
    )
    if thr_info.skip_block and thr_info.holdout_note.startswith("holdout"):
        logger.warning(
            "Bloque descartado por holdout (ev=%.2fbps, señales=%d).",
            holdout_ev_bps,
            thr_info.holdout_signals,
        )

    # Costos dinámicos y compuerta horaria basada en EV por hora
    cost_bps = _cost_bps_from_env()
    evs_train = expected_ev_bps(p_train, TP_BPS, SL_BPS, cost_bps)
    train_hours = Xtr.index.hour.to_numpy()
    test_hours = Xte.index.hour.to_numpy()
    good_hours: set[int] = set()
    min_hour_signals = max(1, HOUR_GATE_MIN_SIGNALS)
    for h in range(24):
        mask = train_hours == h
        count = int(mask.sum())
        if count == 0:
            continue
        ev_mean = float(np.mean(evs_train[mask]))
        if count >= min_hour_signals and ev_mean >= 0.0:
            good_hours.add(h)
    if test_hours.size == 0:
        coverage = 1.0
    else:
        coverage = float(np.mean([h in good_hours for h in test_hours])) if good_hours else 0.0
    if not good_hours or coverage < HOUR_GATE_MIN_COVERAGE:
        logger.debug(
            "Compuerta horaria desactivada (coverage=%.1f%%, horas_buenas=%d).",
            coverage * 100.0,
            len(good_hours),
        )
        hour_ok_series = pd.Series(True, index=Xte.index, dtype=bool)
    else:
        hour_ok = np.array([h in good_hours for h in test_hours], dtype=bool)
        hour_ok_series = pd.Series(hour_ok, index=Xte.index, dtype=bool)

    seg["prob"] = pd.Series(p_test, index=Xte.index)
    seg["signal_note"] = ""

    # EV gating: compute per-row expected EV using dynamic cost estimate
    seg_ev_arr = expected_ev_bps(seg["prob"].values, TP_BPS, SL_BPS, cost_bps)
    seg["ev_bps"] = pd.Series(seg_ev_arr, index=seg.index)

    min_trades_test = max(int(os.getenv("MIN_TRADES_TEST", "8")), 1)

    if EV_ENABLED and not thr_info.skip_block:
        gate_series = seg["ev_bps"] >= EV_MARGIN_BPS
    else:
        gate_series = pd.Series(True, index=seg.index, dtype=bool)

    diag_base = None
    diag_final = None
    signal_state = {
        "trend_mode": "strict",
        "hour_gate_active": False,
        "ev_gate_active": False,
        "threshold_relaxed": False,
        "final_signals": 0,
        "min_trades": min_trades_test,
    }
    relaxations: List[str] = []

    seg["signal"] = pd.Series(0, index=seg.index, dtype=int)
    if thr_info.skip_block:
        reason = thr_info.holdout_note if thr_info.holdout_note else "ev_train_fail"
        logger.warning(
            "Bloque desactivado (%s). EV_train=%.2fbps | señales_train=%d | EV_holdout=%.2fbps | señales_holdout=%d",
            reason,
            thr_info.ev_bps,
            thr_info.signals,
            holdout_ev_bps,
            thr_info.holdout_signals,
        )
    else:
        signal_mask, thr, diag_final, diag_base, signal_state, relaxations = build_signals_with_relaxations(
            seg,
            thr,
            trend,
            gate_series,
            hour_ok_series,
            min_trades=min_trades_test,
            ev_gate_enabled=EV_ENABLED and not thr_info.skip_block,
        )
        seg["signal"] = signal_mask.astype(int)
        if seg["signal"].sum() > 0:
            filters_desc = [
                f"thr={thr:.3f}",
                f"trend={signal_state['trend_mode']}",
                f"hour={'on' if signal_state['hour_gate_active'] else 'off'}",
                f"ev={'on' if signal_state['ev_gate_active'] else 'off'}",
            ]
            if signal_state.get("threshold_relaxed"):
                filters_desc.append("thr_relaxed")
            seg.loc[seg["signal"] == 1, "signal_note"] = ";".join(filters_desc)
        thr_info.threshold = float(thr)

    if diag_base is None:
        diag_base = {"threshold": 0, "after_ev": 0, "after_trend": 0, "after_hours": 0}
    if diag_final is None:
        diag_final = dict(diag_base)

    logger.info(
        "Diagnóstico señales (base thr=%.4f): cand=%d | +EV=%d | trend=%d | hour=%d",
        thr_clamped,
        diag_base["threshold"],
        diag_base["after_ev"],
        diag_base["after_trend"],
        diag_base["after_hours"],
    )
    logger.info(
        "Diagnóstico señales final (thr=%.4f): cand=%d | +EV=%d | trend=%d | hour=%d | finales=%d (mín=%d)",
        thr,
        diag_final["threshold"],
        diag_final["after_ev"],
        diag_final["after_trend"],
        diag_final["after_hours"],
        diag_final["after_hours"],
        min_trades_test,
    )
    if relaxations:
        logger.info("Relajaciones aplicadas: %s", ", ".join(relaxations))

    # Backtest paso-a-paso
    capital = initial_capital
    peak = capital
    in_pos = False
    trades: List[Trade] = []
    eq_times: List[pd.Timestamp] = []
    eq_vals: List[float] = []
    dds: List[float] = []
    trades_started = 0
    trade_cap_logged = False

    idxs = seg.index.to_list()
    bars_in_trade = 0
    consecutive_losses = 0
    # cooldown after a losing trade (in bars)
    COOLDOWN_AFTER_LOSS = int(os.getenv("COOLDOWN_AFTER_LOSS", "2"))
    loss_cooldown = 0
    block_floor = initial_capital * (1.0 - BLOCK_DD_STOP_PCT) if BLOCK_DD_STOP_PCT > 0.0 else -math.inf
    block_stop_triggered = False
    last_time_processed: Optional[pd.Timestamp] = None

    for i in range(len(idxs) - 1):
        t_now = idxs[i]
        t_nxt = idxs[i + 1]
        last_time_processed = t_now

        # decrement cooldown each bar
        if loss_cooldown > 0:
            loss_cooldown -= 1

        close_now = seg.at[t_now, "close"]
        high_nxt = seg.at[t_nxt, "high"]
        low_nxt = seg.at[t_nxt, "low"]
        close_nxt = seg.at[t_nxt, "close"]

        # Mark-to-market (curva)
        eq_times.append(t_now)
        eq_vals.append(capital)
        peak = max(peak, capital)
        dds.append(1.0 - capital / peak)

        if (
            BLOCK_DD_STOP_PCT > 0.0
            and (not in_pos)
            and capital <= block_floor
        ):
            block_stop_triggered = True
            logger.warning(
                "Stop de drawdown intra-bloque activado en %s: capital %.2f <= piso %.2f",
                t_now,
                capital,
                block_floor,
            )
            break

        if in_pos:
            bars_in_trade += 1
            tr = trades[-1]
            tp = tr.take_profit
            sl = tr.stop_loss

            # Gestión temprana (sólo primera barra post-entrada)
            if bars_in_trade == 1:
                # One-bar-fail: si el low cae al 80% del camino a SL, salimos
                fail_level = tr.entry_price - ONE_BAR_FAIL_FRAC * (tr.entry_price - sl)
                if low_nxt <= fail_level:
                    exit_price = fail_level - (close_nxt * SLIPPAGE_PCT)
                    fee_exit = exit_price * tr.size * COMMISSION_RATE
                    pnl = (exit_price - tr.entry_price) * tr.size - fee_exit
                    capital += pnl
                    tr.exit_time = t_nxt
                    tr.exit_price = exit_price
                    tr.commission_exit = fee_exit
                    tr.pnl_usd = tr.realized_usd + pnl
                    tr.pnl_pct = (tr.pnl_usd / (tr.entry_price * tr.size)) * 100.0
                    consecutive_losses = consecutive_losses + 1 if tr.pnl_usd < 0 else 0
                    if tr.pnl_usd < 0:
                        loss_cooldown = COOLDOWN_AFTER_LOSS
                    in_pos = False
                    bars_in_trade = 0
                    continue

                # Early TP (take pequeño en la 1ª barra)
                early_tp = tr.entry_price * bps_to_mult(EARLY_TP_BPS)
                if EARLY_TP_BPS > 0 and high_nxt >= early_tp and bars_in_trade <= EARLY_TP_BARS:
                    # cerrar toda la posición de forma conservadora (simple)
                    exit_price = early_tp - (close_nxt * SLIPPAGE_PCT)
                    fee_exit = exit_price * tr.size * COMMISSION_RATE
                    pnl = (exit_price - tr.entry_price) * tr.size - fee_exit
                    capital += pnl
                    tr.exit_time = t_nxt
                    tr.exit_price = exit_price
                    tr.commission_exit = fee_exit
                    tr.pnl_usd = tr.realized_usd + pnl
                    tr.pnl_pct = (tr.pnl_usd / (tr.entry_price * tr.size)) * 100.0
                    consecutive_losses = consecutive_losses + 1 if tr.pnl_usd < 0 else 0
                    if tr.pnl_usd < 0:
                        loss_cooldown = COOLDOWN_AFTER_LOSS
                    in_pos = False
                    bars_in_trade = 0
                    continue

            # Activar break-even si progresa
            be_level = tr.entry_price + BE_TRIGGER_FRAC * (tp - tr.entry_price)
            if (not tr.be_active) and (high_nxt >= be_level):
                tr.be_active = True
                tr.stop_loss = max(tr.stop_loss, tr.entry_price)  # mover a BE

            # Trailing tras BE
            if tr.be_active:
                trail = close_nxt - TSL_TRAIL_MULT * (tp - tr.entry_price)
                tr.stop_loss = max(tr.stop_loss, trail)

            exit_price = None
            closed = False

            # SL primero (path-aware en barra siguiente)
            if low_nxt <= tr.stop_loss:
                exit_price = tr.stop_loss - (close_nxt * SLIPPAGE_PCT)
                closed = True
            # TP
            if (not closed) and (high_nxt >= tp):
                exit_price = tp + (close_nxt * SLIPPAGE_PCT)
                closed = True
            # Timeout
            if (not closed) and (bars_in_trade >= MAX_BARS_IN_TRADE):
                exit_price = close_nxt - (close_nxt * SLIPPAGE_PCT)
                closed = True

            if closed and exit_price is not None:
                fee_exit = exit_price * tr.size * COMMISSION_RATE
                pnl = (exit_price - tr.entry_price) * tr.size - fee_exit
                capital += pnl

                tr.exit_time = t_nxt
                tr.exit_price = exit_price
                tr.commission_exit = fee_exit
                tr.pnl_usd = tr.realized_usd + pnl
                tr.pnl_pct = (tr.pnl_usd / (tr.entry_price * tr.size)) * 100.0

                consecutive_losses = consecutive_losses + 1 if tr.pnl_usd < 0 else 0
                if tr.pnl_usd < 0:
                    loss_cooldown = COOLDOWN_AFTER_LOSS
                in_pos = False
                bars_in_trade = 0
                if BLOCK_DD_STOP_PCT > 0.0 and capital <= block_floor:
                    block_stop_triggered = True
                    logger.warning(
                        "Stop de drawdown intra-bloque activado tras salida en %s: capital %.2f <= piso %.2f",
                        t_nxt,
                        capital,
                        block_floor,
                    )
                    break
        else:
            # Entrada si hay señal
            if seg.at[t_now, "signal"] == 1:
                if MAX_TRADES_PER_BLOCK > 0 and trades_started >= MAX_TRADES_PER_BLOCK:
                    if not trade_cap_logged:
                        logger.info(
                            "Límite de operaciones por bloque alcanzado (%d). No se abrirán más posiciones hasta el siguiente bloque.",
                            MAX_TRADES_PER_BLOCK,
                        )
                        trade_cap_logged = True
                    continue
                entry = close_now + (close_now * SLIPPAGE_PCT)
                tp = entry * bps_to_mult(TP_BPS)
                sl = entry / bps_to_mult(SL_BPS)
                if sl >= entry:
                    continue
                size = position_size(capital, entry, sl, consecutive_losses)
                if size <= 0:
                    continue
                fee_entry = entry * size * COMMISSION_RATE
                if capital <= fee_entry:
                    continue
                capital -= fee_entry
                prob_now = float(seg.at[t_now, "prob"])
                ev_now = float(seg.at[t_now, "ev_bps"])
                note_now = str(seg.at[t_now, "signal_note"]) if "signal_note" in seg.columns else ""
                trades.append(Trade(
                    entry_time=t_now,
                    entry_price=entry,
                    size=size,
                    stop_loss=sl,
                    take_profit=tp,
                    commission_entry=fee_entry,
                    model_prob=prob_now,
                    model_ev_bps=ev_now,
                    signal_note=note_now,
                ))
                in_pos = True
                bars_in_trade = 0
                trades_started += 1

        if block_stop_triggered:
            break

    # cerrar curva en el último índice
    if len(idxs):
        final_idx = idxs[-1]
        if block_stop_triggered and last_time_processed is not None:
            final_idx = last_time_processed
        if (not eq_times) or eq_times[-1] != final_idx:
            eq_times.append(final_idx)
            eq_vals.append(capital)
            peak = max(peak, capital)
            dds.append(1.0 - capital / peak)

    equity_df = pd.DataFrame({
        "equity_curve": eq_vals,
        "drawdown": dds,
    }, index=pd.Index(eq_times, name="time"))

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    return equity_df, trades_df, capital, thr_info



def train_on_window(df_train: pd.DataFrame) -> Optional[Pipeline]:
    """Entrena un modelo nuevo usando la ventana de entrenamiento."""
    # Preparar features + label 3 velas
    tr = add_features(df_train)
    tr = label_tp_sl_3bar(tr)
    X, y = prep_xy(tr)
    if len(X) < 100:
        return None
    model = build_model()

    # Balanceo simple vía scale_pos_weight
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos > 0 and neg > 0:
        spw = max(neg / pos, 1.0)
        model.named_steps["xgb"].set_params(scale_pos_weight=spw)

    model.fit(X, y)
    return model


def prepare_window(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara una ventana (train o test) con todas las features limpias."""
    win = add_features(df)
    # Asegurar columnas presentes / limpieza
    needed = set(["open","high","low","close","volume"]) | set(FEATURES)
    miss = needed - set(win.columns)
    if miss:
        raise ValueError(f"Faltan columnas en features: {miss}")
    win = handle_missing(win)
    return win


def walk_forward(df: pd.DataFrame, limit_blocks: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Ejecuta la simulación walk-forward ensamblando equity, trades y métricas."""
    start = 0
    capital = INITIAL_CAPITAL
    global_peak = INITIAL_CAPITAL
    all_eq = []
    all_tr = []
    meta = []
    blocks_run = 0

    while True:
        tr_end = start + TRAIN_BARS
        te_end = tr_end + TEST_BARS
        if te_end > len(df):
            logger.info("No hay más datos para WFA. Fin.")
            break

        raw_train = df.iloc[start:tr_end].copy()
        raw_test  = df.iloc[tr_end:te_end].copy()

        # Entrena
        model = train_on_window(raw_train)
        if model is None:
            logger.warning("Ventana con datos insuficientes. Se detiene.")
            break

        # Prepara ventanas con mismas transforms (solo features; label no requerida en test)
        # NOTE: backtest_segment needs a train_df that contains the 3-bar label for threshold
        tr_labeled = add_features(raw_train)
        tr_labeled = label_tp_sl_3bar(tr_labeled)

        tr_feat = prepare_window(raw_train)
        te_feat = prepare_window(raw_test)

        eq, trd, capital, thr_info = backtest_segment(te_feat, model, tr_labeled, capital)

        thr_val = float(thr_info.threshold) if thr_info else float("nan")
        ev_val = float(thr_info.ev_bps) if thr_info else float("nan")
        win_val = float(thr_info.win_rate * 100.0) if thr_info else float("nan")
        logger.info(
            "WFA Train=(%d:%d) Test=(%d:%d) | thr=%.4f | win_train=%.2f%% | ev_train=%.2fbps | trades=%d | capital=%.2f",
            start,
            tr_end,
            tr_end,
            te_end,
            thr_val,
            win_val,
            ev_val,
            len(trd),
            capital,
        )

        test_start_ts = raw_test.index[0].isoformat()
        test_end_ts = raw_test.index[-1].isoformat()
        global_peak = max(global_peak, capital)
        dd_pct = 0.0 if global_peak <= 0 else max(0.0, 1.0 - (capital / global_peak))
        logger.info(
            "[EQUITY] %s → %s equity=$%.2f dd=%.2f%%",
            test_start_ts,
            test_end_ts,
            capital,
            dd_pct * 100.0,
        )

        blocks_run += 1
        if limit_blocks is not None and blocks_run >= int(limit_blocks):
            logger.info(f"--limit-blocks reached: stopping after {blocks_run} blocks")
            break

        if not eq.empty:
            all_eq.append(eq)
        if not trd.empty:
            trd["train_range"] = f"{start}-{tr_end}"
            trd["test_range"] = f"{tr_end}-{te_end}"
            trd["threshold"] = thr_val
            trd["threshold_ev_bps"] = ev_val
            trd["threshold_policy"] = thr_info.policy if thr_info else "unknown"
            all_tr.append(trd)

        meta.append({
            "train_idx": [int(start), int(tr_end)],
            "test_idx": [int(tr_end), int(te_end)],
            "threshold": float(thr_val),
            "threshold_policy": thr_info.policy if thr_info else "unknown",
            "threshold_ev_bps": float(ev_val),
            "threshold_signals": int(thr_info.signals) if thr_info else 0,
            "threshold_skip": bool(thr_info.skip_block) if thr_info else False,
            "trades": int(len(trd)),
            "final_capital": float(capital),
        })

        if capital <= INITIAL_CAPITAL * CAPITAL_STOP_MULTIPLIER:
            logger.warning(
                "Capital cayó a %.2f (%.1f%% del inicial). Stop global activado; se detiene la simulación.",
                capital,
                (capital / INITIAL_CAPITAL) * 100.0,
            )
            break

        start += STEP_BARS

    equity = pd.concat(all_eq) if all_eq else pd.DataFrame(columns=["equity_curve","drawdown"])
    if not equity.empty:
        equity = equity.sort_index()
        cummax = equity["equity_curve"].cummax()
        safe_cummax = cummax.replace(0, np.nan)
        equity["drawdown"] = 1.0 - (equity["equity_curve"] / safe_cummax)
        equity["drawdown"] = equity["drawdown"].fillna(0.0).clip(lower=0.0)
    trades = pd.concat(all_tr, ignore_index=True) if all_tr else pd.DataFrame()

    # Métricas simples
    metrics = {
        "initial_capital": INITIAL_CAPITAL,
        "final_capital": float(equity["equity_curve"].iloc[-1]) if not equity.empty else float(capital),
        "total_return_%": (float(equity["equity_curve"].iloc[-1]) / INITIAL_CAPITAL - 1.0) * 100.0 if not equity.empty else (capital / INITIAL_CAPITAL - 1.0) * 100.0,
        "max_drawdown_%": float((equity["drawdown"].max() * 100.0) if not equity.empty else 0.0),
        "num_trades": int(len(trades)),
        "windows": meta,
        "signal_policy": SIGNAL_POLICY,
        "signal_q": SIGNAL_Q,
        "fixed_threshold": FIXED_THRESHOLD,
        "horizon_bars": HORIZON_BARS,
        "tp_bps": TP_BPS,
        "sl_bps": SL_BPS,
    }

    return equity, trades, metrics



def plot_equity(equity: pd.DataFrame, out_png: str):
    """Genera y guarda el gráfico de curva de capital."""
    if equity.empty:
        logger.warning("Equity vacío; no se grafica.")
        return
    plt.figure(figsize=(11, 6))
    plt.plot(equity.index, equity["equity_curve"], label="Equity")
    plt.title("Equity Curve - WFA (3-bar label)")
    plt.xlabel("Fecha")
    plt.ylabel("Capital")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def ensure_outdir(path: str):
    """Crea el directorio de salida si no existe."""
    os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    """Carga un CSV con OHLCV (espera columna open_time) y normaliza los nombres."""
    if not os.path.exists(path):
        logger.error(f"No existe el archivo: {path}")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["open_time"])  # espera columna open_time
    df = df.set_index("open_time").sort_index()
    cols = {c.lower(): c for c in df.columns}  # soportar mayúsculas
    # Normaliza nombres esperados
    rename_map = {}
    for need in ["open","high","low","close","volume"]:
        if need not in df.columns:
            # intenta variantes
            cand = need.upper()
            if cand in df.columns:
                rename_map[cand] = need
            elif need.capitalize() in df.columns:
                rename_map[need.capitalize()] = need
    if rename_map:
        df = df.rename(columns=rename_map)

    missing = {"open","high","low","close","volume"} - set(df.columns)
    if missing:
        logger.error(f"Faltan columnas obligatorias: {missing}")
        sys.exit(1)
    return df



def main():
    """Punto de entrada CLI del backtest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Ruta al CSV con OHLCV 15m (columna open_time). Si se omite, intenta ML/data/processed/BTCUSDT_15m_processed.csv")
    parser.add_argument("--out", type=str, default=OUTDIR, help="Directorio de salida")
    parser.add_argument("--limit-blocks", type=int, default=None, help="Limita la cantidad de bloques WFA a ejecutar (para pruebas)")
    args = parser.parse_args()

    ensure_outdir(args.out)

    # If --csv not provided, try sensible default path inside the repo
    if args.csv is None:
        default_candidate = os.path.join("ML", "data", "processed", "BTCUSDT_15m_processed.csv")
        if os.path.exists(default_candidate):
            logger.info(f"--csv no proporcionado: usando archivo por defecto {default_candidate}")
            args.csv = default_candidate
        else:
            logger.error(f"--csv no proporcionado y no se encontró el archivo por defecto: {default_candidate}")
            parser.error("--csv es requerido cuando no se encuentra el archivo por defecto")

    df = load_csv(args.csv)
    df = handle_missing(df)

    equity, trades, metrics = walk_forward(df, limit_blocks=args.limit_blocks)

    # Guardar
    equity_path = os.path.join(args.out, "equity_curve.csv")
    trades_path = os.path.join(args.out, "trades.csv")
    metrics_path = os.path.join(args.out, "metrics.json")
    fig_path = os.path.join(args.out, "equity_curve.png")

    equity.to_csv(equity_path)
    trades.to_csv(trades_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_equity(equity, fig_path)

    logger.info(f"Guardado: {equity_path}")
    logger.info(f"Guardado: {trades_path}")
    logger.info(f"Guardado: {metrics_path}")
    logger.info(f"Guardado: {fig_path}")
    logger.info("Listo. Revisa la carpeta de resultados.")


if __name__ == "__main__":
    main()
