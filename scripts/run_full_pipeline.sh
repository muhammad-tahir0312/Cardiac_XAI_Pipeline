#!/bin/bash

# Ensure we are in the project root
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_DIR"

echo "=========================================="
echo "   Cardiac XAI Pipeline Automation        "
echo "=========================================="

# 1. Train Segmentation (UNet)
echo "Step 1: Training Segmentation Model..."
python3 src/train_seg.py --epochs 10

# 2. Train Image Diagnosis (DenseNet)
echo "Step 2: Training DenseNet Classifier..."
python3 src/train_densenet.py --epochs 10

# 3. Train Clinical Classifier (XGBoost)
echo "Step 3: Training Ensemble Classifier..."
python3 src/train_classifier.py

# 4. Generate Predictions and XAI
echo "Step 4: Running Inference and Explanations..."
python3 src/predict.py

echo "=========================================="
echo "   Pipeline Complete! Check /results      "
echo "=========================================="
