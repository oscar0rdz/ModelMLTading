#!/usr/bin/env bash
set -euo pipefail

# Wrapper to clean previous outputs and generate docs/figs + docs/index.md
# Usage:
#   ./scripts/make_pages_clean.sh backtest.log "ML/out/**/trades_block_*.csv"
LOG_FILE="${1:-backtest.log}"
TRADES_GLOB="${2:-ML/out/**/trades_block_*.csv}"

echo "[0/6] Limpieza de resultados previos (si existen)..."
python3 scripts/cleanup_outputs.py --yes || true

echo "[1/6] Preparando carpetas..."
mkdir -p reports docs/figs

echo "[2/6] Extrayendo equity del log..."
python3 scripts/parse_equity_from_logs.py --log "$LOG_FILE" --out reports/equity.csv

echo "[3/6] Generando curva de equity..."
python3 scripts/plot_equity.py --csv reports/equity.csv --out docs/figs/equity_curve.png

echo "[4/6] Figuras extra (PnL hist / razones) desde trades..."
python3 scripts/make_extra_figs.py --glob "$TRADES_GLOB" --outdir docs/figs

echo "[5/6] Métricas y actualización de docs/index.md..."
python3 scripts/metrics_and_index.py \
  --equity_csv reports/equity.csv \
  --trades_glob "$TRADES_GLOB" \
  --index_md docs/index.md

echo "[6/6] Docs listos en docs/ (figs + index.md con métricas)"
