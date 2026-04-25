#!/bin/bash

# Ensure we are in the project root
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_DIR"

# Virtual Environment Path
VENV_PATH="/home/tahir/Automated-Cardiac-Segmentation-and-Disease-Diagnosis/venv"

echo "===================================================="
echo "   Cardiac XAI Pipeline Automation (INTRAC 2026)    "
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

# 1. Train Segmentation (UNet)
echo -e "\n[Step 1/4] Training Segmentation Model (UNet)..."
python src/train_seg.py --epochs 50

# 2. Train Clinical Classifier (XGBoost)
echo -e "\n[Step 2/4] Training Ensemble Classifier (XGBoost)..."
python src/train_classifier.py

# 3. Train Image Diagnosis (DenseNet)
echo -e "\n[Step 3/4] Training DenseNet Classifier..."
python src/train_densenet.py --epochs 20

# 4. Generate Predictions and XAI
echo -e "\n[Step 4/4] Running Inference and Explanations..."
python src/predict.py

echo -e "\n===================================================="
echo "   Pipeline Complete!                              "
echo "   Check the /results folder for models and plots. "
echo "===================================================="
