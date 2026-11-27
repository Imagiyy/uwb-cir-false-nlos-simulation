#!/bin/bash
# run_all.sh — Reproducible runner for the finalized Phantom-NLOS UWB CIR simulation suite.
#
# Usage:
#   ./run_all.sh          # Run full 8-stage simulation, audit, and ablation pipeline
#   ./run_all.sh --core   # Run baseline Stages 1-3 only (inspect, train, floor-wall eval)
#   ./run_all.sh --help   # Show stages and usage

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-all}"
if [[ "$MODE" == "--help" || "$MODE" == "-h" ]]; then
    cat <<'EOF'
Usage: ./run_all.sh [--all | --core | --help]

Stages executed in --all mode (default):
  1/8  src/inspect_data.py              Inspect Fontaine & Santamaria DW3000 datasets
  2/8  src/train.py                     Train clean baseline regressors & classifiers
  3/8  src/evaluate.py                  Floor-Wall sweep (IIoT19 magnitude + IIoT20/DW3000 I/Q)
  4/8  src/diagnose_ae.py               Diagnose AE-Reg (Regime A) OOD bottleneck spike
  5/8  src/eval_insight.py              InSight baselines + low-power sweep + alpha_max bound
  6/8  src/audit_remnet_sub6db.py       Zero-injection check, sub--6 dB sweep, unpadded REMNet-119
  7/8  src/audit_dw3000_native_remnet.py Conditioned zero-shot & disjoint DW3000 proxy-label audit
  8/8  src/ablate_remnet_recipe.py      PyTorch vs. InSight Keras recipe sensitivity ablation

Pass --core to run Stages 1-3 only.
EOF
    exit 0
fi

if [[ "$MODE" != "all" && "$MODE" != "--all" && "$MODE" != "--core" ]]; then
    echo "Unknown option: $MODE (use --help)" >&2
    exit 2
fi

if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
else
    echo "ERROR: Virtual environment not found at .venv/bin/python (and PYTHON not set)." >&2
    echo "Create it with:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

mkdir -p results models

echo ""
echo "========================================================================"
echo "  Phantom-NLOS: Final UWB CIR Simulation & Robustness Evaluation Suite"
echo "========================================================================"
echo ""

echo "=== [1/8] Data Inspection ==="
"$PY" src/inspect_data.py
echo ""

echo "=== [2/8] Clean Model Training (Baseline Regressors & Classifiers) ==="
"$PY" src/train.py
echo ""

echo "=== [3/8] Floor-Wall Sweep (IIoT19 + Complex I/Q IIoT20 & DW3000) ==="
"$PY" src/evaluate.py
echo ""

if [[ "$MODE" == "--core" ]]; then
    echo "=== Core Stages (1-3) Complete ==="
    exit 0
fi

echo "=== [4/8] AE-Reg (Regime A) OOD Bottleneck Diagnosis ==="
"$PY" src/diagnose_ae.py
echo ""

echo "=== [5/8] InSight Baselines, Low-Power Sweep & alpha_max Stealth Bound ==="
"$PY" src/eval_insight.py
echo ""

echo "=== [6/8] Sub--6 dB REMNet & Diffuse LightGBM Audit ==="
"$PY" src/audit_remnet_sub6db.py
echo ""

echo "=== [7/8] Conditioned Zero-Shot & Native DW3000 Audit ==="
"$PY" src/audit_dw3000_native_remnet.py
echo ""

echo "=== [8/8] REMNet Training-Recipe Sensitivity Ablation ==="
"$PY" src/ablate_remnet_recipe.py
echo ""

echo "=== All 8 Simulation Stages Complete ==="
echo "Outputs written under results/."
