import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Common Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

# Segmentation Hyperparameters (UNet)
SEG_BATCH_SIZE = 2
SEG_LR = 5e-4
SEG_EPOCHS = 50
SEG_MODEL_PATH = os.path.join(MODELS_DIR, "unet_model.pth")

# Diagnosis Hyperparameters (DenseNet)
DENSE_BATCH_SIZE = 4
DENSE_LR = 1e-4
DENSE_EPOCHS = 20
DENSE_MODEL_PATH = os.path.join(MODELS_DIR, "densenet_diagnosis.pth")

# Ensemble Diagnosis (Random Forest / XGBoost)
RF_MODEL_PATH = os.path.join(MODELS_DIR, "diagnosis_classifier.joblib")
FEATURE_NAMES = [
    "LV_EDV", "LV_ESV", "LV_EF", 
    "RV_EDV", "RV_ESV", "RV_EF", 
    "Myo_Mass", "LV_RV_Ratio", "Mass_Vol_Ratio"
]

# XAI Settings
SHAP_PLOT_PATH = os.path.join(RESULTS_DIR, "shap_summary.png")
GRADCAM_PLOT_PATH = os.path.join(RESULTS_DIR, "gradcam_result.png")
