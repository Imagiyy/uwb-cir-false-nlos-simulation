#!/bin/bash
# run_all.sh — Single entry point for the Floor-Wall Phantom-NLOS experiment.
#
# Usage:  chmod +x run_all.sh && ./run_all.sh 2>&1 | tee run_all.log
#
# Steps:
#   1. Data inspection (schema + shapes)
#   2. Model training (clean data only)
#   3. Floor-wall evaluation (inject → measure bias & flips)

set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"

# Sanity: check that the venv exists
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Virtual environment not found at $PYTHON"
    echo "Create it with:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

mkdir -p results models

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   Floor-Wall Phantom-NLOS Experiment on UWB CIRs            ║"
echo "║   Hypothesis: ≤+6 dB delayed pulse → >0.5 m bias shift     ║"
echo "║               and/or >70% NLOS classifier flip              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

echo "=== Step 1/3: Data Inspection ==="
echo "    (Verifying dataset schemas, shapes, and column names)"
$PYTHON src/inspect_data.py
echo ""

echo "=== Step 2/3: Training ==="
echo "    (Training on clean data only — LightGBM + APS006; CNN/AE if CIRs available)"
$PYTHON src/train.py
echo ""

echo "=== Step 3/3: Evaluation ==="
echo "    (Floor-wall sweep: inject Phantom-NLOS → measure bias & classifier flips)"
$PYTHON src/evaluate.py
echo ""

echo "=== Experiment Complete ==="
echo "Results saved to:"
echo "  - results/floor_wall_summary.csv"
echo "  - results/bias_shift_vs_power.png"
echo "  - results/nlos_flip_rate_vs_power.png"
echo "  - results/sample_cir_overlay.png"
echo ""

