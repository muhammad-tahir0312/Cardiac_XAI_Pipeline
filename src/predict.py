import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import joblib

import config
from dataloader import ACDC_PatientDataset
from networks import DenseNetDiagnosis
from inference import get_automated_features
from explainers import GradCAM, extract_shap_values_for_class

# ---------------------------------------------------------------------------
# Helper: locate the last denselayer in denseblock4
# ---------------------------------------------------------------------------

def _get_gradcam_layer(model_dense):
    """
    Return the last convolutional sub-module of DenseBlock4 for Grad-CAM.

    """
    try:
        denseblock4 = model_dense.densenet.features.denseblock4
        denselayers = [
            m for name, m in denseblock4.named_children()
            if name.startswith("denselayer")
        ]
        if not denselayers:
            raise AttributeError("No denselayer found in denseblock4.")
        return denselayers[-1]   # last layer is semantically richest
    except AttributeError as e:
        raise RuntimeError(
            f"Cannot locate a suitable Grad-CAM target layer: {e}"
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(patient_idx=0, fold=0):
    device = config.DEVICE

    # ------------------------------------------------------------------ #
    # 1. Load patient data                                                #
    # ------------------------------------------------------------------ #
    dataset = ACDC_PatientDataset(
        root_dir=config.DATA_DIR, fold=fold, mode="val"
    )
    if patient_idx >= len(dataset):
        raise IndexError(
            f"Patient index {patient_idx} out of range "
            f"(fold {fold} val set has {len(dataset)} patients)."
        )
    patient = dataset[patient_idx]
    p_id    = patient["patient_id"]

    print(f"\n{'='*45}")
    print(f"   PIPELINE  —  {p_id}  (fold {fold})")
    print(f"{'='*45}")

    # ------------------------------------------------------------------ #
    # 2. Zero-leakage automated feature extraction                       #
    # ------------------------------------------------------------------ #
    print("Extracting automated clinical features …")
    features = get_automated_features(patient, fold, device)

    # ------------------------------------------------------------------ #
    # 3. Clinical pathway — XGBoost                                      #
    # ------------------------------------------------------------------ #
    clf_path  = config.RF_MODEL_PATH.replace(".joblib", f"_fold{fold}.joblib")
    xgb_probs = np.ones(5) / 5   # uniform fallback

    if os.path.exists(clf_path):
        clf       = joblib.load(clf_path)
        feat_vec  = np.array([[features[k] for k in config.FEATURE_NAMES]])
        xgb_probs = clf.predict_proba(feat_vec)[0]
    else:
        print(f"WARNING: XGBoost model not found at {clf_path}. "
              "Using uniform priors.")

    # ------------------------------------------------------------------ #
    # 4. Visual pathway — DenseNet                                       #
    # ------------------------------------------------------------------ #
    densenet_path = config.DENSE_MODEL_PATH.replace(".pth", f"_fold{fold}.pth")
    dense_probs   = np.ones(5) / 5
    input_t       = None
    ed_s          = None

    if os.path.exists(densenet_path):
        model_dense = DenseNetDiagnosis(class_num=5, pretrained=False).to(device)
        model_dense.load_state_dict(
            torch.load(densenet_path, map_location=device)
        )
        model_dense.eval()

        mid_idx = patient["ed_image"].shape[0] // 2
        ed_s    = patient["ed_image"][mid_idx].astype(np.float32)
        es_s    = patient["es_image"][mid_idx].astype(np.float32)

        def _norm(s):
            return (s - np.mean(s)) / (np.std(s) + 1e-8)

        combined = np.stack([_norm(ed_s), _norm(es_s)], axis=0)  # (2,256,256)
        input_t  = torch.from_numpy(combined).unsqueeze(0).to(device)

        with torch.no_grad():
            output      = model_dense(input_t)
            dense_probs = F.softmax(output, dim=1)[0].cpu().numpy()
    else:
        print(f"WARNING: DenseNet model not found at {densenet_path}. "
              "Using uniform priors.")

    # ------------------------------------------------------------------ #
    # 5. Hybrid ensemble                                                  #
    # ------------------------------------------------------------------ #
    hybrid_probs = 0.6 * xgb_probs + 0.4 * dense_probs
    pred_idx     = int(np.argmax(hybrid_probs))
    inv_map      = config.DIAGNOSIS_MAP_INV
    prediction   = inv_map.get(pred_idx, "UNKNOWN")

    print(f"\n   DIAGNOSTIC REPORT")
    print(f"   XGBoost  (Clinical) : "
          f"{inv_map.get(int(np.argmax(xgb_probs)))}  "
          f"(p={np.max(xgb_probs):.2f})")
    print(f"   DenseNet (Visual)   : "
          f"{inv_map.get(int(np.argmax(dense_probs)))}  "
          f"(p={np.max(dense_probs):.2f})")
    print(f"   CONSENSUS           : {prediction}  "
          f"(p={np.max(hybrid_probs):.2f})")
    if np.max(hybrid_probs) < 0.50:
        print("   *** LOW CONFIDENCE — recommend expert review ***")
    print(f"   Ground Truth        : {patient['diagnosis']}")
    print(f"{'='*45}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 6. Grad-CAM visual explanation                                     #
    # ------------------------------------------------------------------ #
    if input_t is not None and ed_s is not None:
        print("Generating Grad-CAM heatmap …")
        try:
            target_layer = _get_gradcam_layer(model_dense)

            model_dense.eval()
            gcam    = GradCAM(model_dense, target_layer)
            heatmap = gcam.generate_heatmap(input_t, pred_idx)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(ed_s, cmap='gray')
            axes[0].set_title("Original MRI (ED mid-slice)")
            axes[0].axis('off')

            axes[1].imshow(ed_s, cmap='gray')
            axes[1].imshow(heatmap, cmap='jet', alpha=0.5)
            axes[1].set_title(f"Grad-CAM  [{prediction}]")
            axes[1].axis('off')

            plt.suptitle(f"Visual Explanation — {p_id}")
            plt.tight_layout()
            plt.savefig(config.GRADCAM_PLOT_PATH, dpi=150)
            plt.close()
            print(f"Grad-CAM saved to {config.GRADCAM_PLOT_PATH}")
        except Exception as e:
            print(f"WARNING: Grad-CAM failed: {e}")

    # ------------------------------------------------------------------ #
    # 7. SHAP clinical explanation                                       #
    # ------------------------------------------------------------------ #
    if os.path.exists(clf_path):
        print("Generating SHAP clinical justification …")
        try:
            import shap as shap_lib

            clf_loaded  = joblib.load(clf_path)
            explainer   = shap_lib.TreeExplainer(clf_loaded)
            feat_vec    = np.array([[features[k] for k in config.FEATURE_NAMES]])
            shap_values = explainer.shap_values(feat_vec)

            sv = extract_shap_values_for_class(shap_values, pred_idx)

            plt.figure(figsize=(10, 8))
            colors = ['tomato' if v < 0 else 'teal' for v in sv]
            plt.barh(config.FEATURE_NAMES, sv, color=colors)
            plt.axvline(0, color='black', linewidth=0.8)
            plt.xlabel("SHAP Value (contribution to diagnosis)")
            plt.title(
                f"Clinical Justification: {prediction}  (Patient: {p_id})"
            )
            plt.grid(axis='x', linestyle='--', alpha=0.5)
            plt.tight_layout()

            shap_path = os.path.join(config.RESULTS_DIR, f"shap_{p_id}.png")
            plt.savefig(shap_path, dpi=150)
            plt.close()
            print(f"SHAP plot saved to {shap_path}")
        except Exception as e:
            print(f"WARNING: SHAP failed: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hybrid XAI inference pipeline."
    )
    parser.add_argument(
        "--idx",  type=int, default=0,
        help="Patient index within the fold validation set"
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Cross-validation fold (0–4)"
    )
    args = parser.parse_args()
    run_pipeline(args.idx, args.fold)
