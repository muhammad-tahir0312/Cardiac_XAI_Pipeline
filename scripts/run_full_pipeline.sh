#!/bin/bash

# Ensure we are in the project root
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_DIR"

# CRITICAL: Add src to PYTHONPATH so scripts can find dataloader/config/networks
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src

# Virtual Environment Path
VENV_PATH="/home/tahir/Automated-Cardiac-Segmentation-and-Disease-Diagnosis/venv"

echo "===================================================="
echo "   Cardiac XAI Pipeline: 5-FOLD K-FOLD SYSTEM       "
echo "   Venue: INTRAC 2026 | UM FSKTM                    "
echo "===================================================="

# Check if virtual environment is active, if not, try to activate it
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [ -f "$VENV_PATH/bin/activate" ]; then
        echo "Activating virtual environment..."
        source "$VENV_PATH/bin/activate"
    else
        echo "WARNING: Virtual environment not found at $VENV_PATH."
        echo "Proceeding with system python (ensure dependencies are installed)."
    fi
fi

# 1. Train Segmentation (5 Folds)
echo -e "\n[Step 1/3] Training 5-Fold UNet Segmentation Models..."
python3 src/train_seg.py --epochs 30

# 2. Train Ensemble Classifier (5 Folds)
echo -e "\n[Step 2/3] Training 5-Fold XGBoost Classifiers (Clinical Pathway)..."
echo "Note: This step extracts features from predicted masks (Zero Leakage)."
python3 src/train_classifier.py

# 3. Train Image Diagnosis (5 Folds)
echo -e "\n[Step 3/3] Training 5-Fold Multi-View DenseNet Models..."
python3 src/train_densenet.py --epochs 30

# 4. Generate Predictions and XAI
echo -e "\n[Inference] Running Final Pipeline (Fold 0, Patient 0)..."
python3 src/predict.py --idx 0 --fold 0

echo -e "\n===================================================="
echo "   Pipeline Complete!                              "
echo "   Models saved as: model_name_foldX.pth           "
echo "   Check the /results folder for evaluation plots. "
echo "===================================================="
