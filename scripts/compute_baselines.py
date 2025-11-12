#!/usr/bin/env python3
"""
Compute risk metrics for the ML strategy vs. simple baselines (buy & hold, naive momentum).
Outputs both CSV and Markdown tables in results_wfa/.
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


BAR_MINUTES_DEFAULT = 15
COMMISSION_SIDE = 0.0003  # same cost used in backtest per side
ROUND_TRIP_COST = 2 * COMMISSION_SIDE


@dataclass
class EquitySeries:
    times: List[datetime]
    values: List[float]


def parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    value = value.replace(" ", "T")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None


def read_equity(path: Path) -> EquitySeries:
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        times, values = [], []
        for row in reader:
            val = row.get("equity_curve") or row.get("equity_usd") or row.get("equity")
            t_raw = row.get("time") or row.get("timestamp")
            if val is None:
                continue
            times.append(parse_datetime(t_raw) if t_raw else None)
            values.append(float(val))
    return EquitySeries(times=times, values=values)


def read_trades(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def calc_drawdown(equity: Sequence[float]) -> Tuple[float, float]:
    peak = -math.inf
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        if peak <= 0:
            continue
        dd = (val - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return max_dd, abs(max_dd) * 100.0


def calc_returns(series: Sequence[float]) -> List[float]:
    returns = []
    for prev, curr in zip(series[:-1], series[1:]):
        if prev == 0:
            continue
        returns.append((curr - prev) / prev)
    return returns


def calc_minutes_per_step(times: Sequence[Optional[datetime]], default_minutes: float = BAR_MINUTES_DEFAULT) -> float:
    deltas = []
    prev = None
    for t in times:
        if t is None or prev is None:
            prev = t
            continue
        delta_min = (t - prev).total_seconds() / 60.0
        if delta_min > 0:
            deltas.append(delta_min)
        prev = t
    if not deltas:
        return default_minutes
    return sum(deltas) / len(deltas)


def calc_sharpe(returns: Sequence[float], minutes_per_step: float) -> float:
    if len(returns) < 2:
        return float("nan")
    try:
        mu = statistics.mean(returns)
        sigma = statistics.stdev(returns)
    except statistics.StatisticsError:
        return float("nan")
    if sigma == 0:
        return float("nan")
    bars_per_day = (24 * 60) / minutes_per_step if minutes_per_step > 0 else 96
    annual_factor = math.sqrt(365 * bars_per_day)
    return (mu / sigma) * annual_factor


def calc_calmar(equity: Sequence[float], duration_days: float, max_dd_decimal: float) -> float:
    if not equity or duration_days <= 0 or max_dd_decimal >= 0:
        return float("nan")
    years = duration_days / 365.0
    if years <= 0:
        return float("nan")
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1 if equity[0] > 0 else float("nan")
    if not math.isfinite(cagr) or max_dd_decimal == 0:
        return float("nan")
    return cagr / abs(max_dd_decimal)


def duration_days(times: Sequence[Optional[datetime]]) -> float:
    valid = [t for t in times if t is not None]
    if len(valid) < 2:
        return max(len(times) / 96.0, 1.0)
    span = (max(valid) - min(valid)).total_seconds() / 86400.0
    return max(span, 1.0)


def build_price_series(trades: List[Dict[str, str]]) -> List[Tuple[datetime, float]]:
    buckets: Dict[datetime, List[float]] = defaultdict(list)
    for row in trades:
        for key, price_key in (("entry_time", "entry_price"), ("exit_time", "exit_price")):
            t = parse_datetime(row.get(key, ""))
            price_raw = row.get(price_key)
            if t is None or price_raw is None or price_raw == "":
                continue
            try:
                buckets[t].append(float(price_raw))
            except ValueError:
                continue
    entries = []
    for t, prices in buckets.items():
        entries.append((t, sum(prices) / len(prices)))
    entries.sort(key=lambda x: x[0])
    return entries


def equity_from_prices(prices: List[Tuple[datetime, float]], initial_capital: float) -> EquitySeries:
    if not prices:
        return EquitySeries(times=[], values=[])
    first_price = prices[0][1]
    values = []
    for _, price in prices:
        values.append(initial_capital * (price / first_price))
    times = [ts for ts, _ in prices]
    return EquitySeries(times=times, values=values)


def buy_hold_metrics(price_series: List[Tuple[datetime, float]], initial_capital: float) -> Dict[str, float]:
    eq = equity_from_prices(price_series, initial_capital)
    if not eq.values:
        return {"pnl": 0.0, "maxdd_pct": float("nan"), "sharpe": float("nan"),
                "calmar": float("nan"), "winrate": float("nan"), "trades_per_day": 0.0}
    returns = calc_returns(eq.values)
    minutes_step = calc_minutes_per_step(eq.times)
    max_dd_decimal, max_dd_pct = calc_drawdown(eq.values)
    dur_days = duration_days(eq.times)
    sharpe = calc_sharpe(returns, minutes_step)
    calmar = calc_calmar(eq.values, dur_days, max_dd_decimal)
    winrate = (sum(1 for r in returns if r > 0) / len(returns) * 100) if returns else float("nan")
    trade_return = (eq.values[-1] - eq.values[0]) / eq.values[0] if eq.values[0] else 0.0
    trades_per_day = 1.0 / dur_days
    return {
        "pnl": eq.values[-1] - eq.values[0],
        "maxdd_pct": max_dd_pct,
        "sharpe": sharpe,
        "calmar": calmar,
        "winrate": winrate,
        "trades_per_day": trades_per_day,
        "trade_returns": [trade_return - ROUND_TRIP_COST]
    }


def naive_momentum_metrics(price_series: List[Tuple[datetime, float]], initial_capital: float,
                           lookback: int = 12) -> Dict[str, float]:
    if len(price_series) <= lookback:
        return {"pnl": 0.0, "maxdd_pct": float("nan"), "sharpe": float("nan"),
                "calmar": float("nan"), "winrate": float("nan"), "trades_per_day": 0.0, "trade_returns": []}
    capital = initial_capital
    equity_values = [capital]
    position = 0
    entry_price = None
    trade_returns: List[float] = []
    sma_window: List[float] = []
    trade_open_time: Optional[datetime] = None

    for idx in range(1, len(price_series)):
        prev_price = price_series[idx - 1][1]
        curr_price = price_series[idx][1]
        sma_window.append(prev_price)
        if len(sma_window) > lookback:
            sma_window.pop(0)
        signal = 1 if len(sma_window) == lookback and prev_price > (sum(sma_window) / lookback) else 0

        if signal == 1 and position == 0:
            position = 1
            entry_price = prev_price
            trade_open_time = price_series[idx - 1][0]
            capital *= (1 - COMMISSION_SIDE)
        elif signal == 0 and position == 1:
            if entry_price is not None:
                gross = (prev_price - entry_price) / entry_price
                trade_returns.append(gross - ROUND_TRIP_COST)
            position = 0
            entry_price = None
            trade_open_time = None
            capital *= (1 - COMMISSION_SIDE)

        ret = (curr_price - prev_price) / prev_price if prev_price else 0.0
        capital *= (1 + ret * position)
        equity_values.append(capital)

    if position == 1 and entry_price is not None:
        last_price = price_series[-1][1]
        gross = (last_price - entry_price) / entry_price
        trade_returns.append(gross - ROUND_TRIP_COST)
        capital *= (1 - COMMISSION_SIDE)
        equity_values[-1] = capital

    returns = calc_returns(equity_values)
    minutes_step = calc_minutes_per_step([ts for ts, _ in price_series])
    max_dd_decimal, max_dd_pct = calc_drawdown(equity_values)
    dur_days = duration_days([ts for ts, _ in price_series])
    sharpe = calc_sharpe(returns, minutes_step)
    calmar = calc_calmar(equity_values, dur_days, max_dd_decimal)
    winrate = (sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100) if trade_returns else float("nan")
    trades_per_day = (len(trade_returns) / dur_days) if dur_days else 0.0
    return {
        "pnl": equity_values[-1] - equity_values[0],
        "maxdd_pct": max_dd_pct,
        "sharpe": sharpe,
        "calmar": calmar,
        "winrate": winrate,
        "trades_per_day": trades_per_day,
        "trade_returns": trade_returns
    }


def ml_strategy_metrics(equity: EquitySeries, trades: List[Dict[str, str]]) -> Dict[str, float]:
    values = equity.values
    minutes_step = calc_minutes_per_step(equity.times)
    ret_series = calc_returns(values)
    max_dd_decimal, max_dd_pct = calc_drawdown(values)
    dur_days = duration_days(equity.times)
    sharpe = calc_sharpe(ret_series, minutes_step)
    calmar = calc_calmar(values, dur_days, max_dd_decimal)
    trade_returns = []
    trade_times = []
    for row in trades:
        pnl_pct = row.get("pnl_pct")
        entry_time = parse_datetime(row.get("entry_time", ""))
        if pnl_pct is not None and pnl_pct != "":
            try:
                trade_returns.append(float(pnl_pct))
            except ValueError:
                continue
        if entry_time:
            trade_times.append(entry_time)
    winrate = (sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100) if trade_returns else float("nan")
    duration_from_trades = duration_days(trade_times) if trade_times else dur_days
    trades_per_day = (len(trade_returns) / duration_from_trades) if duration_from_trades else 0.0
    return {
        "pnl": values[-1] - values[0] if values else 0.0,
        "maxdd_pct": max_dd_pct,
        "sharpe": sharpe,
        "calmar": calmar,
        "winrate": winrate,
        "trades_per_day": trades_per_day,
        "trade_returns": trade_returns
    }


def format_value(value: float, decimals: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "N/D"
    return f"{value:.{decimals}f}"


def write_tables(rows: List[Dict[str, str]], csv_path: Path, md_path: Path) -> None:
    fieldnames = ["Strategy", "PnL (USD)", "MaxDD (%)", "Sharpe", "Calmar", "WinRate (%)", "Trades/Día"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(fieldnames) + " |\n")
        fh.write("|" + "|".join(["---"] * len(fieldnames)) + "|\n")
        for row in rows:
            fh.write("| " + " | ".join(row[col] for col in fieldnames) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Compute baseline vs ML WFA metrics.")
    parser.add_argument("--equity_csv", default="results_wfa/equity_curve.csv")
    parser.add_argument("--trades_csv", default="results_wfa/trades.csv")
    parser.add_argument("--initial_capital", type=float, default=1000.0)
    parser.add_argument("--out_csv", default="results_wfa/baselines_risk.csv")
    parser.add_argument("--out_md", default="results_wfa/baselines_risk.md")
    args = parser.parse_args()

    equity = read_equity(Path(args.equity_csv))
    trades = read_trades(Path(args.trades_csv))
    price_series = build_price_series(trades)

    ml_metrics = ml_strategy_metrics(equity, trades)
    buy_hold = buy_hold_metrics(price_series, args.initial_capital)
    naive = naive_momentum_metrics(price_series, args.initial_capital)

    rows = []
    rows.append({
        "Strategy": "ML · EV=0.582 (WFA)",
        "PnL (USD)": format_value(ml_metrics["pnl"]),
        "MaxDD (%)": format_value(ml_metrics["maxdd_pct"]),
        "Sharpe": format_value(ml_metrics["sharpe"], 3),
        "Calmar": format_value(ml_metrics["calmar"], 3),
        "WinRate (%)": format_value(ml_metrics["winrate"]),
        "Trades/Día": format_value(ml_metrics["trades_per_day"], 2),
    })
    rows.append({
        "Strategy": "Buy & Hold (BTC)",
        "PnL (USD)": format_value(buy_hold["pnl"]),
        "MaxDD (%)": format_value(buy_hold["maxdd_pct"]),
        "Sharpe": format_value(buy_hold["sharpe"], 3),
        "Calmar": format_value(buy_hold["calmar"], 3),
        "WinRate (%)": format_value(buy_hold["winrate"]),
        "Trades/Día": format_value(buy_hold["trades_per_day"], 2),
    })
    rows.append({
        "Strategy": "Naive Momentum (SMA-12)",
        "PnL (USD)": format_value(naive["pnl"]),
        "MaxDD (%)": format_value(naive["maxdd_pct"]),
        "Sharpe": format_value(naive["sharpe"], 3),
        "Calmar": format_value(naive["calmar"], 3),
        "WinRate (%)": format_value(naive["winrate"]),
        "Trades/Día": format_value(naive["trades_per_day"], 2),
    })

    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_tables(rows, csv_path, Path(args.out_md))
    print(f"[OK] Tabla de baselines guardada en {csv_path} y {args.out_md}")


if __name__ == "__main__":
    main()
