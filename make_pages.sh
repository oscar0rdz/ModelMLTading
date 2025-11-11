#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   ./make_pages.sh backtest.log "ML/out/**/trades_block_*.csv"
LOG_FILE="${1:-backtest.log}"
TRADES_GLOB="${2:-ML/out/**/trades_block_*.csv}"

echo "[1/5] Preparando carpetas..."
mkdir -p reports docs/figs

echo "[2/5] Extrayendo equity del log..."
python scripts/parse_equity_from_logs.py --log "$LOG_FILE" --out reports/equity.csv

echo "[3/5] Generando curva de equity..."
python scripts/plot_equity.py --csv reports/equity.csv --out docs/figs/equity_curve.png

echo "[4/5] Figuras extra (PnL hist / razones) desde trades..."
python scripts/make_extra_figs.py --glob "$TRADES_GLOB" --outdir docs/figs

echo "[5/5] Métricas y actualización de docs/index.md..."
python scripts/metrics_and_index.py \
  --equity_csv reports/equity.csv \
  --trades_glob "$TRADES_GLOB" \
  --index_md docs/index.md

echo "[OK] Docs listos en docs/ (figs + index.md con métricas)"
