import torch
import torch.nn.functional as F
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from dataloader import ACDC_PatientDataset
from networks import UNet, DenseNetDiagnosis
from inference import predict_full_volume, get_automated_features, post_process_mask
from explainers import GradCAM
import config
import shap

def run_pipeline(patient_idx=0, fold=0):
    device = config.DEVICE
    
    # 1. Load Data (Using Fold system to ensure zero leakage)
    dataset = ACDC_PatientDataset(root_dir=config.DATA_DIR, fold=fold, mode="val")
    patient = dataset[patient_idx]
    p_id = patient["patient_id"]
    print(f"\n" + "="*40)
    print(f"   SOTA PIPELINE: {p_id} (FOLD {fold})")
    print(f"="*40)
    
    # 2. Automated Feature Extraction (NO LEAKAGE)
    print("Extracting automated clinical features (predicting volume)...")
    features = get_automated_features(patient, fold, device)
    
    # 3. Hybrid Ensemble Prediction
    # XGBoost Pathway
    clf_path = config.RF_MODEL_PATH.replace(".joblib", f"_fold{fold}.joblib")
    xgb_probs = np.zeros(5)
    if os.path.exists(clf_path):
        clf = joblib.load(clf_path)
        feat_vec = np.array([[features[k] for k in config.FEATURE_NAMES]])
        xgb_probs = clf.predict_proba(feat_vec)[0]
    
    # DenseNet Pathway
    densenet_path = config.DENSE_MODEL_PATH.replace(".pth", f"_fold{fold}.pth")
    dense_probs = np.zeros(5)
    if os.path.exists(densenet_path):
        model_dense = DenseNetDiagnosis(class_num=5, pretrained=False).to(device)
        model_dense.load_state_dict(torch.load(densenet_path, map_location=device))
        model_dense.eval()
        
        mid_idx = patient["ed_image"].shape[0] // 2
        ed_s = patient["ed_image"][mid_idx].astype(np.float32)
        es_s = patient["es_image"][mid_idx].astype(np.float32)
        
        def norm(s): return (s - np.mean(s)) / (np.std(s) + 1e-8)
        input_combined = np.stack([norm(ed_s), norm(es_s)], axis=0)
        input_t = torch.from_numpy(input_combined).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model_dense(input_t)
            dense_probs = F.softmax(output, dim=1)[0].cpu().numpy()

    # Consensus Logic
    hybrid_probs = 0.6 * xgb_probs + 0.4 * dense_probs
    pred_idx = np.argmax(hybrid_probs)
    class_map_inv = config.DIAGNOSIS_MAP_INV
    prediction = class_map_inv.get(pred_idx, "UNKNOWN")
    
    print(f"\n   REPORT")
    print(f"XGBoost (Clinical): {class_map_inv.get(np.argmax(xgb_probs))} (Prob: {np.max(xgb_probs):.2f})")
    print(f"DenseNet (Image):  {class_map_inv.get(np.argmax(dense_probs))} (Prob: {np.max(dense_probs):.2f})")
    print(f"FINAL CONSENSUS:   {prediction} (Prob: {np.max(hybrid_probs):.2f})")
    print(f"TRUE DIAGNOSIS:    {patient['diagnosis']}")
    print(f"="*40)

    # 4. Explainability (XAI)
    # 4a. Visual XAI: Grad-CAM
    if os.path.exists(densenet_path):
        print("Generating Grad-CAM focus map...")
        target_layer = model_dense.densenet.features.denseblock4.denselayer16
        gcam = GradCAM(model_dense, target_layer)
        heatmap = gcam.generate_heatmap(input_t, pred_idx)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(ed_s, cmap='gray'); plt.title("Original MRI (ED)")
        plt.subplot(1, 2, 2); plt.imshow(ed_s, cmap='gray'); plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.savefig(config.GRADCAM_PLOT_PATH)
        plt.close()

    # 4b. Clinical XAI: SHAP
    if os.path.exists(clf_path):
        print("Generating SHAP clinical justification...")
        explainer = shap.TreeExplainer(clf)
        feat_vec = np.array([[features[k] for k in config.FEATURE_NAMES]])
        shap_values = explainer.shap_values(feat_vec)
        
        plt.figure(figsize=(10, 6))
        # Handle binary vs multiclass SHAP output format
        sv = shap_values[pred_idx][0] if isinstance(shap_values, list) else shap_values[0]
        
        plt.barh(config.FEATURE_NAMES, sv, color='teal')
        plt.xlabel("SHAP Value (Contribution to Diagnosis)")
        plt.title(f"Clinical Justification for {prediction} ({p_id})")
        plt.tight_layout()
        shap_plot_path = os.path.join(config.RESULTS_DIR, f"shap_{p_id}.png")
        plt.savefig(shap_plot_path)
        plt.close()
        print(f"SHAP clinical plot saved to {shap_plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Patient index")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold")
    args = parser.parse_args()
    run_pipeline(args.idx, args.fold)
