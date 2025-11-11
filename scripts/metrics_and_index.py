#!/usr/bin/env python3
import argparse, os, glob, re, pandas as pd, numpy as np
from datetime import datetime

def max_drawdown(equity):
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd.min()  # negativo

def load_trades(trades_glob):
    files = glob.glob(trades_glob, recursive=True)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def compute_metrics(equity_csv, trades_glob):
    # Equity y MaxDD
    eq = pd.read_csv(equity_csv)
    if "equity_usd" not in eq.columns:
        raise ValueError("equity_csv debe tener columna 'equity_usd'")
    total_pnl_usd = eq["equity_usd"].astype(float).iloc[-1] - eq["equity_usd"].astype(float).iloc[0]
    mdd = max_drawdown(eq["equity_usd"].astype(float))  # negativo
    mdd_pct = round(abs(mdd) * 100, 2) if pd.notna(mdd) else None

    # Trades (si existen)
    trades = load_trades(trades_glob)
    trades_total = None
    winrate = None
    avg_net_bps = None

    if not trades.empty:
        trades_total = int(len(trades))
        # detectar columna PnL
        cols = {c.lower(): c for c in trades.columns}
        pnl_col = None
        for k in ["pnl_usd","pnl","net_usd","net"]:
            if k in cols:
                pnl_col = cols[k]; break
        if pnl_col:
            s = pd.to_numeric(trades[pnl_col], errors="coerce")
            winrate = round(float((s > 0).mean() * 100), 2)

        # detectar columna net_bps
        bps_col = None
        for k in ["net_bps","ev_bps","ret_bps"]:
            if k in cols:
                bps_col = cols[k]; break
        if bps_col:
            b = pd.to_numeric(trades[bps_col], errors="coerce")
            avg_net_bps = round(float(np.nanmean(b)), 4)

    return {
        "pnl_total_usd": round(float(total_pnl_usd), 2),
        "maxdd_pct": mdd_pct,
        "trades_total": trades_total,
        "winrate_pct": winrate,
        "avg_net_bps": avg_net_bps
    }

def ensure_index(index_md):
    if os.path.exists(index_md):
        return
    os.makedirs(os.path.dirname(index_md), exist_ok=True)
    with open(index_md, "w", encoding="utf-8") as f:
        f.write(
"""
# ModelMLTrading — Walk-Forward Analysis (BTCUSDT 15m)

**Resumen**
Señales ML para BTC/USDT (15m). Pipeline: preproceso → entrenamiento → selección por EV (bps) → WFA → backtest (dynamic/label).

## Resultados visuales
### Equity curve
![Equity curve](./figs/equity_curve.png)

### Distribución de PnL por trade
![PnL histogram](./figs/pnl_hist.png)

### Razones de salida
![Razones](./figs/reasons.png)

## Métricas clave
<!--METRICS:START-->
Pendiente de calcular.
<!--METRICS:END-->

## Cómo reproducir
```bash
python ML/backtest_improved.py > backtest.log
./make_pages.sh backtest.log "ML/out/**/trades_block_*.csv"
```
"""
        )

def update_metrics_in_index(index_md, metrics):
    ensure_index(index_md)
    with open(index_md, "r", encoding="utf-8") as f:
        txt = f.read()

    # Build metrics block
    lines = ["<!--METRICS:START-->\n"]
    if metrics.get("trades_total") is not None:
        lines.append(f"Trades totales: {metrics['trades_total']}\n")
    lines.append(f"PnL total (USD): {metrics['pnl_total_usd']}\n")
    if metrics.get("winrate_pct") is not None:
        lines.append(f"Win-rate: {metrics['winrate_pct']}%\n")
    lines.append(f"Max Drawdown (%): {metrics.get('maxdd_pct') if metrics.get('maxdd_pct') is not None else 'N/D'}\n")
    if metrics.get("avg_net_bps") is not None:
        lines.append(f"Avg net bps/trade: {metrics['avg_net_bps']}\n")
    else:
        lines.append("Avg net bps/trade: N/D\n")
    lines.append("<!--METRICS:END-->")
    new_block = "".join(lines)

    if "<!--METRICS:START-->" in txt and "<!--METRICS:END-->" in txt:
        txt = re.sub(r"<!--METRICS:START-->.*?<!--METRICS:END-->", new_block, txt, flags=re.S)
    else:
        txt += "\n\n" + new_block + "\n"

    with open(index_md, "w", encoding="utf-8") as f:
        f.write(txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity_csv", required=True)
    ap.add_argument("--trades_glob", required=True)
    ap.add_argument("--index_md", required=True)
    args = ap.parse_args()

    m = compute_metrics(args.equity_csv, args.trades_glob)
    update_metrics_in_index(args.index_md, m)
    print("[OK] Métricas calculadas e insertadas en", args.index_md)

if __name__ == "__main__":
    main()
