import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Segmentation  (UNet)
# ---------------------------------------------------------------------------
SEG_BATCH_SIZE = 4
SEG_LR         = 5e-4
SEG_EPOCHS     = 50
SEG_MODEL_PATH = os.path.join(MODELS_DIR, "unet_model.pth")

# ---------------------------------------------------------------------------
# Visual diagnosis  (DenseNet-121)
# ---------------------------------------------------------------------------
DENSE_BATCH_SIZE = 4
DENSE_LR         = 1e-4
DENSE_EPOCHS     = 50
DENSE_MODEL_PATH = os.path.join(MODELS_DIR, "densenet_diagnosis.pth")

# ---------------------------------------------------------------------------
# Clinical classifier  (XGBoost)
# ---------------------------------------------------------------------------
RF_MODEL_PATH = os.path.join(MODELS_DIR, "diagnosis_classifier.joblib")

# 18 clinical features extracted from automated segmentation masks
FEATURE_NAMES = [
    "LV_EDV", "LV_ESV", "LV_SV", "LV_EF",
    "RV_EDV", "RV_ESV", "RV_SV", "RV_EF",
    "Myo_Mass", "LV_RV_Ratio", "Mass_Vol_Ratio",
    "RWT", "LVEDVi", "LVESVi", "SVi", "Massi",
    "Myo_Thickening", "RV_FAC",
]

# ---------------------------------------------------------------------------
# XAI output paths
# ---------------------------------------------------------------------------
SHAP_PLOT_PATH    = os.path.join(RESULTS_DIR, "shap_summary.png")
GRADCAM_PLOT_PATH = os.path.join(RESULTS_DIR, "gradcam_result.png")

# ---------------------------------------------------------------------------
# Diagnosis classes  (alphabetical — consistent with sklearn LabelEncoder)
# ---------------------------------------------------------------------------
DIAGNOSIS_CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]
DIAGNOSIS_MAP     = {cls: i for i, cls in enumerate(DIAGNOSIS_CLASSES)}
DIAGNOSIS_MAP_INV = {i: cls for i, cls in enumerate(DIAGNOSIS_CLASSES)}
