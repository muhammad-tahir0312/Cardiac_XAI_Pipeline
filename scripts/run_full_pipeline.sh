#!/bin/bash
# =============================================================================
#  Cardiac XAI Pipeline — Full 5-Fold Training + Inference
#  Venue : INTRAC 2026  |  Universiti Malaya FSKTM
#  Author: Muhammad Tahir student in FSKTM, UM
# =============================================================================

set -euo pipefail   # exit on error, undefined variable, or pipe failure

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
SRC_DIR="$PROJECT_DIR/src"
VENV_PATH="$PROJECT_DIR/venv"

echo "===================================================="
echo "   Cardiac XAI Pipeline: 5-FOLD CROSS-VALIDATION   "
echo "   Venue: INTRAC 2026 | UM FSKTM                    "
echo "   Project root : $PROJECT_DIR"
echo "   Source dir   : $SRC_DIR"
echo "===================================================="

if [[ "${VIRTUAL_ENV:-}" == "" ]]; then
    if [ -f "$VENV_PATH/bin/activate" ]; then
        echo "Activating virtual environment at $VENV_PATH …"
        # shellcheck disable=SC1091
        source "$VENV_PATH/bin/activate"
    else
        echo "WARNING: Virtual environment not found at $VENV_PATH."
        echo "Proceeding with system Python (ensure all dependencies are installed)."
    fi
fi

cd "$PROJECT_DIR"

export PYTHONPATH="${PYTHONPATH:-}:$SRC_DIR"

# --------------------------------------------------------------------------- #
# Step 1 — Segmentation (UNet, 5 folds)                                       #
# --------------------------------------------------------------------------- #
echo ""
echo "[Step 1/4] Training 5-Fold UNet Segmentation Models …"
python3 "$SRC_DIR/train_seg.py" --epochs 30

# --------------------------------------------------------------------------- #
# Step 2 — Clinical Classifier (XGBoost, 5 folds, zero-leakage)              #
# --------------------------------------------------------------------------- #
echo ""
echo "[Step 2/4] Training 5-Fold XGBoost Classifiers …"
echo "           Features derived from predicted masks (zero-leakage)."
python3 "$SRC_DIR/train_classifier.py"

# --------------------------------------------------------------------------- #
# Step 3 — Visual Classifier (DenseNet-121, 5 folds)                         #
# --------------------------------------------------------------------------- #
echo ""
echo "[Step 3/4] Training 5-Fold DenseNet-121 Models …"
python3 "$SRC_DIR/train_densenet.py" --epochs 30

# --------------------------------------------------------------------------- #
# Step 4 — Inference + XAI for fold 0, patient 0                             #
# --------------------------------------------------------------------------- #
echo ""
echo "[Step 4/4] Running pipeline (fold=0, patient_idx=0) …"
python3 "$SRC_DIR/predict.py" --idx 0 --fold 0

echo ""
echo "===================================================="
echo "   Pipeline complete!"
echo "   Models  → $PROJECT_DIR/models/"
echo "   Results → $PROJECT_DIR/results/"
echo "===================================================="
