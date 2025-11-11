#!/usr/bin/env python3
import argparse, glob, os, pandas as pd, numpy as np, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="ML/out/**/trades_block_*.csv")
    ap.add_argument("--outdir", default="docs/figs")
    args = ap.parse_args()

    files = glob.glob(args.glob, recursive=True)
    os.makedirs(args.outdir, exist_ok=True)

    if not files:
        # Placeholders si aún no hay CSVs de trades
        open(os.path.join(args.outdir, "pnl_hist.png"), "wb").close()
        open(os.path.join(args.outdir, "reasons.png"), "wb").close()
        print("Sin CSVs de trades; creados placeholders.")
        return

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass
    if not dfs:
        open(os.path.join(args.outdir, "pnl_hist.png"), "wb").close()
        open(os.path.join(args.outdir, "reasons.png"), "wb").close()
        print("CSV de trades ilegibles; creados placeholders.")
        return

    df = pd.concat(dfs, ignore_index=True)
    cols = {c.lower(): c for c in df.columns}

    # Histograma de PnL
    pnl_col = None
    for k in ["pnl_usd","pnl","net_usd","net"]:
        if k in cols: pnl_col = cols[k]; break
    if pnl_col:
        s = pd.to_numeric(df[pnl_col], errors="coerce").dropna()
        plt.figure(figsize=(8,4.5))
        plt.hist(s, bins=50)
        plt.title("Distribución de PnL por trade")
        plt.xlabel("PnL (USD)"); plt.ylabel("Frecuencia")
        plt.grid(True, linewidth=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"pnl_hist.png"), dpi=144)
        plt.close()
    else:
        open(os.path.join(args.outdir, "pnl_hist.png"), "wb").close()

    # Razones de salida
    reason_col = None
    for k in ["reason","exit_reason","close_reason"]:
        if k in cols: reason_col = cols[k]; break
    if reason_col:
        c = df[reason_col].astype(str).value_counts().sort_values(ascending=False)
        plt.figure(figsize=(8,4.5))
        c.plot(kind="bar")
        plt.title("Razones de salida (conteo)")
        plt.xlabel("Razón"); plt.ylabel("Trades")
        plt.grid(True, axis="y", linewidth=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"reasons.png"), dpi=144)
        plt.close()
    else:
        open(os.path.join(args.outdir, "reasons.png"), "wb").close()

    print(f"OK: figuras -> {args.outdir}/(pnl_hist.png, reasons.png)")

if __name__ == "__main__":
    main()
