# Cardiac XAI Pipeline: Automated Diagnosis & Interpretability

This project implements a state-of-the-art pipeline for **Automated Cardiac Diagnosis** from MRI scans. It leverages deep learning for segmentation and disease classification, integrated with **Explainable AI (XAI)** frameworks (SHAP and Grad-CAM) for clinical transparency.

## 🚀 Quick Start

### 1. Activate Environment
Always run the pipeline within the provided virtual environment:
```bash
source /home/tahir/Automated-Cardiac-Segmentation-and-Disease-Diagnosis/venv/bin/activate
```

### 2. Run Full Pipeline
To run the entire training and inference process:
```bash
bash scripts/run_full_pipeline.sh
```
# 1. Train the Ensemble Diagnosis model (Fast)
python src/train_classifier.py

# 2. Train the Segmentation UNet (Slow - let it run)
python src/train_seg.py

# 3. Train the Deep Learning Diagnosis model (Slow - let it run)
python src/train_densenet.py

python src/predict.py

---

## 🛠️ Pipeline Architecture

1.  **Segmentation (UNet)**: A high-capacity PyTorch UNet segments the Left Ventricle (LV), Right Ventricle (RV), and Myocardium. Now enhanced with **Z-score normalization** and increased feature channels.
2.  **Clinical Feature Engineering**:
    *   **BSA Indexing**: All volumes are indexed by **Body Surface Area (BSA)** calculated from patient height/weight.
    *   **New Metrics**: Includes **Relative Wall Thickness (RWT)** and Stroke Volume Index (**SVi**).
3.  **Diagnosis Models**:
    *   **Ensemble (XGBoost)**: Optimized with `RandomizedSearchCV` on clinical features.
    *   **Deep Learning (DenseNet121)**: Trained on MRI slices with class-weighted loss.
4.  **Explainability (XAI)**:
    *   **SHAP**: Quantifies clinical feature contribution to each diagnosis.
    *   **Grad-CAM**: Provides heatmaps of neural network focus on MRI slices.

## 📁 Project Structure

```text
src/
├── dataloader.py       # ACDC dataset loader with BSA & Z-score normalization
├── networks.py         # UNet and DenseNet architectures (Enhanced)
├── extract_features.py # Clinical metric calculations (Indexed by BSA)
├── train_seg.py        # Segmentation training with ReduceLROnPlateau
├── train_classifier.py # XGBoost training with hyperparameter tuning
├── train_densenet.py   # DenseNet training with class weights
├── explainers.py       # SHAP and Grad-CAM implementations
└── predict.py          # End-to-end inference & visualization
results/                # Saved models, metrics, and XAI visualizations
```

## 📊 Evaluation
Results are saved to the `results/` folder:
- `seg_history.json`: Training history for segmentation.
- `seg_viz_*.png`: Visual proof of segmentation accuracy.
- `gradcam_result.png`: Explainability focus map.
- `clinical_distributions.png`: Statistical separation of cardiac diseases.
- `confusion_matrix.png`: Diagnostic performance metrics.

---
*Designed for the **INTRAC 2026** Transdisciplinary Conference.*
