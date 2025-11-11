#!/usr/bin/env python3
import argparse, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV con: timestamp,equity_usd,dd_pct")
    ap.add_argument("--out", required=True, help="PNG de salida, ej. docs/figs/equity_curve.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "timestamp" in df.columns:
        try: df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception: pass

    plt.figure(figsize=(10,5))
    if "timestamp" in df.columns:
        plt.plot(df["timestamp"], df["equity_usd"], linewidth=1.8)
        plt.xlabel("Time")
    else:
        plt.plot(df["equity_usd"], linewidth=1.8)
        plt.xlabel("Index")
    plt.title("Equity Curve")
    plt.ylabel("Equity (USD)")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=144)
    print(f"OK: figura -> {args.out}")

if __name__ == "__main__":
    main()
