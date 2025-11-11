# -*- coding: utf-8 -*-
"""
wfa_retrain_block.py — Bloque drop-in para WFA con reentrenamiento por ventana,
calibración isotónica y selección de umbral por EV con piso (EV_MEAN_MIN).

Uso mínimo (en backtest_improved.py):
from ML.wfa_retrain_block import (
    load_env_cfg, prefer_data_path, fit_xgb_with_cv,
    calibrate_isotonic,
    iter_wfa_windows, apply_filters_and_gate, equity_engine
)
from ML.ev_selector import select_threshold_by_ev_unificado
cfg = load_env_cfg()
path = prefer_data_path(cfg)
# ... carga tu DF procesado en df (con features) ...
for tr_idx, te_idx, meta in iter_wfa_windows(df.index, cfg):
    Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
    Xte, yte = X.iloc[te_idx], y.iloc[te_idx]
    model, _ = fit_xgb_with_cv(Xtr, ytr, cfg)
    iso = calibrate_isotonic(model, Xtr, ytr, holdout_frac=0.15)
    p_raw = model.predict_proba(Xte)[:,1]
    p_cal = iso.transform(p_raw.reshape(-1,1)).ravel() if iso is not None else p_raw
    thr, stats = select_threshold_by_ev_unificado(p_cal, yte.to_numpy())
    adm = apply_filters_and_gate(df.iloc[te_idx], p_cal, thr, cfg)
    equity, summary = equity_engine(adm, cfg)
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import Iterator, Tuple, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from ML.ev_selector import select_threshold_by_ev_unificado


# Config

@dataclass
class Cfg:
    # Paths
    DATA_PATH: str
    ALT_DATA_PATH: str
    RESULTS_WFA_DIR: str

    # Label/horizonte
    LOOK_AHEAD: int

    # Costes & EV
    COMMISSION_RATE: float
    SPREAD_PCT: float
    SLIPPAGE_BASE: float
    SLIP_RANGE_COEF: float
    SLIP_MAX_PCT: float
    TP_MULT: float
    SL_MULT: float
    EV_MEAN_MIN: float
    EV_MARGIN: float
    EV_GRID_LOW: float
    EV_GRID_HIGH: float
    EV_GRID_POINTS: int
    MIN_TP_TO_COST: float
    MIN_TRADES_DAILY: float
    BARS_PER_DAY: int

    # Banda no-trade
    NO_TRADE_BAND_LOW: float
    NO_TRADE_BAND_HIGH: float

    # Filtros de régimen
    ADX_MIN: float
    MIN_RANGE_C_ENTRY: float

    # Riesgo y control
    INITIAL_CAPITAL: float
    LEVERAGE: float
    RISK_PER_TRADE: float
    DRAWDOWN_LIMIT: float
    DAILY_LOSS_LIMIT_PCT: float
    ENABLE_SHORTS: int
    MAX_TRADES_PER_DAY: int
    MAX_HOLD_BARS: int
    COOLDOWN_BARS: int

    # Selección de candidatos
    TOPK_DAILY: int
    MIN_RAW_CANDS_PER_DAY: int

    # WFA
    WFA_TRAIN_DAYS: int
    WFA_TEST_DAYS: int
    WFA_STEP_DAYS: int

    # Entrenamiento ligero para WFA
    N_TRIALS_LOCAL: int
    SAMPLE_FRAC: float


def _getenv(key: str, default, cast):
    v = os.getenv(key, default)
    try:
        return cast(v)
    except Exception:
        return cast(default)


def load_env_cfg() -> Cfg:
    """Lee variables de entorno y arma la configuración con defaults seguros."""
    return Cfg(
        # paths
        DATA_PATH=os.getenv("DATA_PATH", "ML/data/processed/BTCUSDT_15m_processed.csv"),
        ALT_DATA_PATH=os.getenv("ALT_DATA_PATH", "ML/data/BTCUSDT_15m_processed.csv"),
        RESULTS_WFA_DIR=os.getenv("RESULTS_WFA_DIR", "ML/results_wfa"),

        # label/look
        LOOK_AHEAD=_getenv("LOOK_AHEAD", 3, int),

        # costes & EV
        COMMISSION_RATE=_getenv("COMMISSION_RATE", 0.00030, float),
        SPREAD_PCT=_getenv("SPREAD_PCT", 0.00005, float),
        SLIPPAGE_BASE=_getenv("SLIPPAGE_BASE", 0.00010, float),
        SLIP_RANGE_COEF=_getenv("SLIP_RANGE_COEF", 0.10, float),
        SLIP_MAX_PCT=_getenv("SLIP_MAX_PCT", 0.00100, float),
        TP_MULT=_getenv("TP_MULT", 1.15, float),
        SL_MULT=_getenv("SL_MULT", 0.60, float),
        EV_MEAN_MIN=_getenv("EV_MEAN_MIN", 0.015, float),
        EV_MARGIN=_getenv("EV_MARGIN", 0.010, float),
        EV_GRID_LOW=_getenv("EV_GRID_LOW", 0.70, float),
        EV_GRID_HIGH=_getenv("EV_GRID_HIGH", 0.98, float),
        EV_GRID_POINTS=_getenv("EV_GRID_POINTS", 25, int),
        MIN_TP_TO_COST=_getenv("MIN_TP_TO_COST", 0.80, float),
        MIN_TRADES_DAILY=_getenv("MIN_TRADES_DAILY", 0.40, float),
        BARS_PER_DAY=_getenv("BARS_PER_DAY", 96, int),

        # banda no-trade
        NO_TRADE_BAND_LOW=_getenv("NO_TRADE_BAND_LOW", 0.48, float),
        NO_TRADE_BAND_HIGH=_getenv("NO_TRADE_BAND_HIGH", 0.52, float),

        # filtros
        ADX_MIN=_getenv("ADX_MIN", 25, float),
        MIN_RANGE_C_ENTRY=_getenv("MIN_RANGE_C_ENTRY", 0.0030, float),

        # riesgo
        INITIAL_CAPITAL=_getenv("INITIAL_CAPITAL", 500.0, float),
        LEVERAGE=_getenv("LEVERAGE", 1.0, float),
        RISK_PER_TRADE=_getenv("RISK_PER_TRADE", 0.003, float),
        DRAWDOWN_LIMIT=_getenv("DRAWDOWN_LIMIT", 0.15, float),
        DAILY_LOSS_LIMIT_PCT=_getenv("DAILY_LOSS_LIMIT_PCT", 0.01, float),
        ENABLE_SHORTS=_getenv("ENABLE_SHORTS", 0, int),
        MAX_TRADES_PER_DAY=_getenv("MAX_TRADES_PER_DAY", 1, int),
        MAX_HOLD_BARS=_getenv("MAX_HOLD_BARS", 3, int),
        COOLDOWN_BARS=_getenv("COOLDOWN_BARS", 10, int),

        # selección
        TOPK_DAILY=_getenv("TOPK_DAILY", 1, int),
        MIN_RAW_CANDS_PER_DAY=_getenv("MIN_RAW_CANDS_PER_DAY", 0, int),

        # WFA
        WFA_TRAIN_DAYS=_getenv("WFA_TRAIN_DAYS", 360, int),
        WFA_TEST_DAYS=_getenv("WFA_TEST_DAYS", 30, int),
        WFA_STEP_DAYS=_getenv("WFA_STEP_DAYS", 30, int),

        # entrenamiento ligero
        N_TRIALS_LOCAL=_getenv("N_TRIALS_LOCAL", 8, int),
        SAMPLE_FRAC=_getenv("SAMPLE_FRAC", 0.70, float),
    )


def prefer_data_path(cfg: Cfg) -> str:
    """Elige DATA_PATH si existe; si no, ALT_DATA_PATH."""
    if os.path.exists(cfg.DATA_PATH):
        return cfg.DATA_PATH
    if os.path.exists(cfg.ALT_DATA_PATH):
        return cfg.ALT_DATA_PATH
    raise FileNotFoundError(
        f"No se encontró DATA_PATH ni ALT_DATA_PATH: {cfg.DATA_PATH} | {cfg.ALT_DATA_PATH}"
    )


# Entrenamiento ligero (CV)

def fit_xgb_with_cv(X: pd.DataFrame, y: pd.Series, cfg: Cfg):
    """
    Entrena un XGBClassifier con CV temporal (grid pequeño) usando PR-AUC.
    Acelera con SAMPLE_FRAC y usa 3 configuraciones razonables.
    Devuelve (modelo_entrenado, {'pr_auc_cv': valor}).
    """
    if cfg.SAMPLE_FRAC < 1.0:
        m = int(len(X) * cfg.SAMPLE_FRAC)
        X, y = X.iloc[-m:], y.iloc[-m:]

    tscv = TimeSeriesSplit(n_splits=4)
    grid = [
        dict(max_depth=6,  learning_rate=0.03,  n_estimators=400, subsample=0.7,  colsample_bytree=0.8,  reg_alpha=0.2,  reg_lambda=1.0, gamma=0.8),
        dict(max_depth=8,  learning_rate=0.02,  n_estimators=350, subsample=0.65, colsample_bytree=0.85, reg_alpha=1.2,  reg_lambda=0.8, gamma=1.6),
        dict(max_depth=10, learning_rate=0.018, n_estimators=330, subsample=0.60, colsample_bytree=0.82, reg_alpha=1.1,  reg_lambda=0.7, gamma=2.0),
    ]

    best_pr, best_model = -1.0, None
    for p in grid[: max(1, cfg.N_TRIALS_LOCAL)]:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",  # solo como métrica interna de XGB
            n_jobs=0,
            tree_method="hist",
            **p
        )
        pr_scores = []
        for tr, va in tscv.split(X):
            model.fit(X.iloc[tr], y.iloc[tr])
            proba = model.predict_proba(X.iloc[va])[:, 1]
            precision, recall, _ = precision_recall_curve(y.iloc[va], proba)
            pr_auc = float(np.trapz(precision, recall))
            pr_scores.append(pr_auc)
        score = float(np.mean(pr_scores))
        if score > best_pr:
            best_pr, best_model = score, model

    best_model.fit(X, y)
    return best_model, {"pr_auc_cv": best_pr}


# Calibración isotónica (hold-out)

def calibrate_isotonic(model, X_train: pd.DataFrame, y_train: pd.Series, holdout_frac=0.15):
    """
    Calibra con IsotonicRegression en el tramo final del train (hold-out),
    y reentrena el modelo en todo el train después.
    """
    n = len(X_train)
    h = max(100, int(n * holdout_frac))
    X_fit, y_fit = X_train.iloc[:-h], y_train.iloc[:-h]
    X_hold, y_hold = X_train.iloc[-h:], y_train.iloc[-h:]

    model.fit(X_fit, y_fit)
    p_hold = model.predict_proba(X_hold)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_hold, y_hold.astype(float))

    model.fit(X_train, y_train)  # reentrena completo para uso final
    return iso


# Umbral por EV (envoltorio unificado)

def select_threshold_by_ev(p_up: np.ndarray, y_true: np.ndarray, cfg: Cfg, idx_times=None):
    """
    Wrapper que reutiliza el selector unificado y respeta la configuración
    del entorno (trades/día objetivo y barras por día).
    """
    thr, stats = select_threshold_by_ev_unificado(
        p_up,
        y_true,
        target_trades_per_day=cfg.MIN_TRADES_DAILY,
        bars_per_day=cfg.BARS_PER_DAY,
        idx_times=idx_times,
    )
    return thr, stats


# Ventanas WFA por días

def iter_wfa_windows(index: pd.DatetimeIndex, cfg: Cfg) -> Iterator[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Genera índices (train_idx, test_idx) con ventanas en días:
      train = WFA_TRAIN_DAYS, test = WFA_TEST_DAYS, step = WFA_STEP_DAYS.
    """
    ts = pd.Series(range(len(index)), index=index).resample("1D").last().dropna()
    days = ts.index
    for start in range(0, len(days) - (cfg.WFA_TRAIN_DAYS + cfg.WFA_TEST_DAYS) + 1, cfg.WFA_STEP_DAYS):
        train_from = days[start]
        train_to   = days[start + cfg.WFA_TRAIN_DAYS - 1]
        test_from  = days[start + cfg.WFA_TRAIN_DAYS]
        test_to    = days[start + cfg.WFA_TRAIN_DAYS + cfg.WFA_TEST_DAYS - 1]

        tr_mask = (index >= train_from) & (index <= train_to)
        te_mask = (index >= test_from) & (index <= test_to)

        meta = dict(
            train_from=str(train_from), train_to=str(train_to),
            test_from=str(test_from),   test_to=str(test_to)
        )
        yield np.where(tr_mask)[0], np.where(te_mask)[0], meta


# Filtros, gating y ejecución

def apply_filters_and_gate(df_slice: pd.DataFrame, p_up: np.ndarray, thr: float, cfg: Cfg) -> pd.DataFrame:
    """
    Aplica:
      - Banda no-trade (probabilidades ambiguas).
      - Filtro ADX mínimo y rango relativo mínimo (si existen columnas).
      - Umbral por EV final.
    Devuelve DataFrame con columnas: p_up, admit(bool).
    """
    out = pd.DataFrame(index=df_slice.index)
    out["p_up"] = p_up.astype(float)
    out["admit"] = True

    # banda no-trade
    amb = (out["p_up"] >= cfg.NO_TRADE_BAND_LOW) & (out["p_up"] <= cfg.NO_TRADE_BAND_HIGH)
    out.loc[amb, "admit"] = False

    # filtros si existen
    if "ADX_14" in df_slice.columns:
        out.loc[df_slice["ADX_14"] < cfg.ADX_MIN, "admit"] = False
    if "range_c" in df_slice.columns:
        out.loc[df_slice["range_c"] < cfg.MIN_RANGE_C_ENTRY, "admit"] = False

    # umbral
    out.loc[out["p_up"] < thr, "admit"] = False

    # TOPK_DAILY y MIN_RAW_CANDS_PER_DAY podrían aplicarse aquí por día si se quiere
    return out


def equity_engine(adm: pd.DataFrame, cfg: Cfg):
    """
    Motor de equity simple:
      - Arriesga RISK_PER_TRADE * capital por trade admitido.
      - P&L proxy usando el "edge" (p_up relativo a 0.5) menos costes.
      - Respeta MAX_TRADES_PER_DAY, COOLDOWN_BARS, DAILY_LOSS_LIMIT_PCT y DRAWDOWN_LIMIT.
    Nota: Sustitúyelo por tu motor real si ya simulas TP/SL/HOLD a LOOK_AHEAD.
    """
    capital = cfg.INITIAL_CAPITAL
    dd_floor = capital * (1 - cfg.DRAWDOWN_LIMIT)
    daily_stop = cfg.DAILY_LOSS_LIMIT_PCT

    eq = []
    cooldown = 0
    day = None
    trades_today = 0
    daily_pl = 0.0

    # costos “por trade” en R aprox
    cost_R = (cfg.COMMISSION_RATE * 2.0) + cfg.SPREAD_PCT + cfg.SLIPPAGE_BASE

    for ts, row in adm.iterrows():
        # rollover de día
        if (day is None) or (ts.date() != day):
            day = ts.date()
            trades_today = 0
            daily_pl = 0.0

        # entrar si cumple gating y límites
        if bool(row.get("admit", False)) and (trades_today < cfg.MAX_TRADES_PER_DAY) and (cooldown == 0):
            risk_amt = capital * cfg.RISK_PER_TRADE
            edge = max(-1.0, min(1.0, (float(row["p_up"]) - 0.5) * 2.0))
            pnl = risk_amt * (edge - cost_R)  # proxy
            capital += pnl
            daily_pl += pnl
            trades_today += 1
            cooldown = cfg.COOLDOWN_BARS

        if cooldown > 0:
            cooldown -= 1

        # paros
        if capital <= dd_floor:
            eq.append((ts, capital))
            break
        if daily_pl <= -abs(capital * daily_stop):
            cooldown = cfg.COOLDOWN_BARS  # freno intradía

        eq.append((ts, capital))

    equity = pd.Series([c for _, c in eq], index=[t for t, _ in eq], name="equity")
    summary = dict(
        final=float(equity.iloc[-1] if len(equity) else capital),
        return_total_pct=float(((equity.iloc[-1] / cfg.INITIAL_CAPITAL) - 1) * 100.0 if len(equity) else 0.0)
    )
    return equity, summary
