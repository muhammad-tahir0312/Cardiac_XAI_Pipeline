source /home/tahir/Automated-Cardiac-Segmentation-and-Disease-Diagnosis/venv/bin/activate

bash scripts/run_full_pipeline.sh

# 1. Train the Ensemble Diagnosis model (Fast)
python src/train_classifier.py

# 2. Train the Segmentation UNet (Slow - let it run)
python src/train_seg.py

# 3. Train the Deep Learning Diagnosis model (Slow - let it run)
python src/train_densenet.py

python src/predict.py


# Cardiac XAI Pipeline

This project implements a complete pipeline for Automated Cardiac Diagnosis from MRI scans using Explainable AI (XAI) frameworks (SHAP and Grad-CAM). It is designed for the INTRAC 2026 transdisciplinary conference.

## Architecture

1.  **Segmentation:** PyTorch UNet (adapted from SSL4MIS) segments the Left Ventricle, Right Ventricle, and Myocardium.
2.  **Feature Extraction:** Clinical metrics (EF, EDV, ESV, Mass) are extracted from the generated masks.
3.  **Diagnosis:**
    *   **Ensemble Classifier:** A Random Forest trained on clinical features (High Accuracy & Interpretability).
    *   **Deep Classifier:** A DenseNet121 trained directly on MRI slices.
4.  **Explainability:**
    *   **SHAP:** Justifies the diagnosis based on clinical features.
    *   **Grad-CAM:** Visualizes where the neural network is focusing on the MRI scans.

## Project Structure

```text
src/
├── dataloader.py       # ACDC dataset loader (SimpleITK)
├── networks.py         # UNet and DenseNet architectures
├── extract_features.py # Clinical metric calculations
├── train_seg.py        # UNet segmentation training
├── train_classifier.py # Random Forest diagnosis training
├── train_densenet.py   # DenseNet diagnosis training
├── explainers.py       # SHAP and Grad-CAM implementations
└── predict.py          # End-to-end pipeline execution
results/                # Saved models and XAI visualizations
data/                   # Symlinked ACDC dataset
```

## How to Run

### 1. Requirements
Ensure you are using the provided virtual environment or have the following installed:
`torch`, `torchvision`, `SimpleITK`, `numpy`, `scikit-learn`, `shap`, `matplotlib`, `joblib`, `tqdm`.

### 2. Training
Run the training scripts in order:
```bash
python src/train_seg.py         # Train Segmentation
python src/train_classifier.py  # Train Ensemble Diagnosis
python src/train_densenet.py    # Train Deep Diagnosis
```

### 3. Prediction & XAI
Run the end-to-end prediction and visualization:
```bash
python src/predict.py
```

## Results
- Check `results/` for the trained models (`.pth` and `.joblib`).
- Check `results/shap_summary.png` for the clinical feature importance chart.
