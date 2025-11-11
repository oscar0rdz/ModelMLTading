#!/bin/bash
# This script runs the complete ML pipeline: data processing, model training, and backtesting.

echo "--- Step 1: Data Processing ---"
python -m ML.data_processing

echo "--- Step 2: Model Training ---"
python -m ML.model_training

echo "--- Step 3: Walk-Forward Backtest ---"
python -m ML.backtest_improved

echo "--- Pipeline Finished ---"
