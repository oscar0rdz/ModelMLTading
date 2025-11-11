# -*- coding: utf-8 -*-
"""
Selector unificado de umbral por EV.

Maximiza EV_total sujeto a un piso de EV_mean, mínimo de señales y
ritmo objetivo de operaciones por día. Considera reward:risk (TP/SL)
y costes expresados en múltiplos de riesgo (R).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def _ensure_array(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def select_threshold_by_ev_unificado(
    p_up: Iterable[float],
    y_true: Iterable[float],
    *,
    ev_grid_low: float | None = None,
    ev_grid_high: float | None = None,
    ev_grid_points: int | None = None,
    ev_mean_min: float | None = None,
    ev_margin_pct: float | None = None,
    tp_mult: float | None = None,
    sl_mult: float | None = None,
    commission: float | None = None,
    spread: float | None = None,
    slippage: float | None = None,
    min_signals: int | None = None,
    target_trades_per_day: float | None = None,
    bars_per_day: int | None = None,
    idx_times: Iterable[Any] | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Selecciona un único umbral que maximiza EV_total, exige EV_mean
    mínimo y aproxima el ritmo de operaciones al objetivo deseado.
    """

    p_up = _ensure_array(p_up)
    y_true = _ensure_array(y_true)

    ev_grid_low = float(os.getenv("EV_GRID_LOW", ev_grid_low or 0.55))
    ev_grid_high = float(os.getenv("EV_GRID_HIGH", ev_grid_high or 0.80))
    ev_grid_points = int(os.getenv("EV_GRID_POINTS", ev_grid_points or 50))
    ev_mean_min = float(os.getenv("EV_MEAN_MIN", ev_mean_min or 0.015))
    ev_margin_pct = float(os.getenv("EV_MARGIN_PCT", ev_margin_pct or 0.005))
    tp_mult = float(os.getenv("TP_MULT", tp_mult or 1.15))
    sl_mult = float(os.getenv("SL_MULT", sl_mult or 0.60))
    commission = float(os.getenv("COMMISSION_RATE", commission or 0.00030))
    spread = float(os.getenv("SPREAD_PCT", spread or 0.00005))
    slippage = float(os.getenv("SLIPPAGE_BASE", slippage or 0.00010))
    min_signals = int(os.getenv("MIN_SIGNALS_EV", min_signals or 90))
    target_trades_per_day = target_trades_per_day or float(os.getenv("MIN_TRADES_DAILY", 0.40))
    bars_per_day = bars_per_day or int(os.getenv("BARS_PER_DAY", 96))

    rr = max(0.1, tp_mult / max(1e-8, sl_mult))
    cost_r = commission * 2 + spread + slippage

    qs = np.linspace(ev_grid_low, ev_grid_high, ev_grid_points)
    thr_candidates = np.unique(np.quantile(p_up, qs))
    approx_days = None
    if idx_times is not None:
        try:
            idx_series = pd.to_datetime(list(idx_times))
            approx_days = max(1.0, float(len(idx_series.normalize().unique())))
        except Exception:
            approx_days = None
    if approx_days is None:
        n_bars = len(p_up)
        approx_days = max(1.0, n_bars / max(1, bars_per_day))

    best: Tuple[float, float, float, int, float, float] | None = None
    candidates = []

    for thr in thr_candidates:
        thr = float(thr)
        mask = p_up >= thr
        n = int(mask.sum())
        pwin = float(np.mean(y_true[mask])) if n else 0.0
        trades_per_day = n / approx_days if approx_days > 0 else 0.0

        reason = "accepted"
        ev_mean = float("nan")
        ev_total = float("-inf")

        if n < min_signals:
            reason = "n<min_signals"
        else:
            ev_mean = pwin * rr - (1.0 - pwin) - cost_r
            # use ev_margin_pct (percentage margin) provided via args or env
            if not np.isfinite(ev_mean) or ev_mean < (ev_mean_min + ev_margin_pct):
                reason = "ev_mean<piso"
            else:
                ev_total = ev_mean * n
                gap = abs(trades_per_day - target_trades_per_day) if target_trades_per_day else 0.0
                cand = (ev_total, ev_mean, thr, n, gap, trades_per_day)
                if (best is None) or (cand[1] > best[1]) or (abs(cand[1] - best[1]) < 0.005 and cand[4] < best[4]):
                    best = cand
                reason = "selected" if best is not None and cand[:4] == best[:4] else "accepted"

        candidates.append(
            {
                "thr": float(thr),
                "n": int(n),
                "pwin": float(pwin),
                "ev_mean": float(ev_mean) if np.isfinite(ev_mean) else None,
                "ev_total": float(ev_total) if np.isfinite(ev_total) else None,
                "trades_per_day": float(trades_per_day),
                "reason": reason,
            }
        )

    if best is None:
        thr_fallback = float(os.getenv("THR_BASE_DEFAULT", 0.78))
        stats = {
            "ev_total": 0.0,
            "ev_mean": 0.0,
            "n": 0,
            "RR": float(rr),
            "cost_R": float(cost_r),
            "trades_per_day": 0.0,
            "note": "fallback: no candidate met constraints",
            "candidates": candidates,
        }
        return thr_fallback, stats

    ev_total, ev_mean, thr, n, gap, trades_per_day = best
    stats = {
        "ev_total": float(ev_total),
        "ev_mean": float(ev_mean),
        "n": int(n),
        "RR": float(rr),
        "cost_R": float(cost_r),
        "trades_per_day": float(trades_per_day),
        "candidates": candidates,
    }
    return float(thr), stats
