#!/usr/bin/env python3
import argparse, pandas as pd, glob, os, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="ML/out/**/trades_block_*.csv")
    ap.add_argument("--out", default="reports/trades_summary.csv")
    args = ap.parse_args()

    files = glob.glob(args.glob, recursive=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        pnl_col = None
        for k in ["pnl_usd","pnl","net_usd","net"]:
            if k in cols: pnl_col = cols[k]; break
        winrate = avg_pnl = None
        if pnl_col:
            s = pd.to_numeric(df[pnl_col], errors="coerce")
            winrate = (s > 0).mean()
            avg_pnl = float(np.nanmean(s))
        rows.append({
            "file": f,
            "trades": len(df),
            "winrate_pct": None if winrate is None else round(100*winrate,2),
            "avg_pnl_usd": None if avg_pnl is None or np.isnan(avg_pnl) else round(avg_pnl,6)
        })
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"OK: resumen -> {args.out} ({len(rows)} archivos)")

if __name__ == "__main__":
    main()
