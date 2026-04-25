import torch
import torch.nn.functional as F
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from dataloader import ACDC_PatientDataset
from networks import UNet, DenseNetDiagnosis
from extract_features import extract_patient_features
from explainers import GradCAM, explain_diagnosis_shap
from visualize import visualize_segmentation, visualize_feature_maps
import config
from scipy.ndimage import label

def post_process_mask(mask):
    """
    SOTA Trick: Keep only the largest connected component for each foreground class.
    This removes small false positive blobs.
    """
    refined_mask = np.zeros_like(mask)
    for c in range(1, 4): # Labels 1, 2, 3
        class_mask = (mask == c)
        labeled_array, num_features = label(class_mask)
        if num_features > 0:
            # Find the largest component
            largest_cc = 0
            max_size = 0
            for i in range(1, num_features + 1):
                size = np.sum(labeled_array == i)
                if size > max_size:
                    max_size = size
                    largest_cc = i
            refined_mask[labeled_array == largest_cc] = c
    return refined_mask

def run_pipeline(patient_idx=0):
    device = config.DEVICE
    results_dir = config.RESULTS_DIR
    
    # 1. Load Data
    dataset = ACDC_PatientDataset(root_dir=config.DATA_DIR, split="validation_set")
    patient = dataset[patient_idx]
    p_id = patient["patient_id"]
    print(f"Running SOTA Pipeline for {p_id}...")
    
    # 2. Segmentation with TTA (Test-Time Augmentation)
    model_seg = UNet(in_chns=1, class_num=4).to(device)
    if os.path.exists(config.SEG_MODEL_PATH):
        model_seg.load_state_dict(torch.load(config.SEG_MODEL_PATH, map_location=device))
        model_seg.eval()
        
        mid = len(patient["ed_mask"]) // 2
        mri_slice = patient["ed_image"][mid].astype(np.float32)
        # Z-score Normalization
        mri_slice_norm = (mri_slice - np.mean(mri_slice)) / (np.std(mri_slice) + 1e-8)
        
        # TTA: Original, H-Flip, V-Flip
        from dataloader import ACDC_Dataset
        dummy_ds = ACDC_Dataset.__new__(ACDC_Dataset)
        input_img = dummy_ds._pad_or_crop(mri_slice_norm, (256, 256))
        
        input_t = torch.from_numpy(input_img).unsqueeze(0).unsqueeze(0).to(device)
        input_hf = torch.flip(input_t, dims=[3])
        input_vf = torch.flip(input_t, dims=[2])
        
        with torch.no_grad():
            out_orig = F.softmax(model_seg(input_t), dim=1)
            out_hf = torch.flip(F.softmax(model_seg(input_hf), dim=1), dims=[3])
            out_vf = torch.flip(F.softmax(model_seg(input_vf), dim=1), dims=[2])
            
            # Average probabilities
            out_avg = (out_orig + out_hf + out_vf) / 3.0
            pred = torch.argmax(out_avg, dim=1)[0].cpu().numpy()
            
        # Post-processing
        pred_mask = post_process_mask(pred)
        gt_mask = patient["ed_mask"][mid]
        visualize_segmentation(mri_slice, gt_mask, pred_mask, save_name=f"seg_viz_{p_id}.png")
    else:
        print("Segmentation model not found.")

    # 3. Hybrid Diagnosis Ensemble (XGBoost + DenseNet)
    # Extract SOTA Features
    features = extract_patient_features(patient["ed_mask"], patient["es_mask"], patient["spacing"], bsa=patient["bsa"])
    
    # XGBoost Prediction (Clinical)
    clf_path = config.RF_MODEL_PATH
    xgb_probs = np.zeros(5)
    if os.path.exists(clf_path):
        clf = joblib.load(clf_path)
        feat_vec = np.array([[features[k] for k in config.FEATURE_NAMES]])
        xgb_probs = clf.predict_proba(feat_vec)[0]
    
    # DenseNet Prediction (Image-based)
    densenet_path = config.DENSE_MODEL_PATH
    dense_probs = np.zeros(5)
    if os.path.exists(densenet_path):
        model_dense = DenseNetDiagnosis(class_num=5, pretrained=False).to(device)
        model_dense.load_state_dict(torch.load(densenet_path, map_location=device))
        model_dense.eval()
        
        # Mid-slices pair
        mid_idx = len(patient["ed_image"]) // 2
        ed_s = patient["ed_image"][mid_idx].astype(np.float32)
        es_s = patient["es_image"][mid_idx].astype(np.float32)
        
        def norm(s): return (s - np.mean(s)) / (np.std(s) + 1e-8)
        input_combined = np.stack([norm(ed_s), norm(es_s)], axis=0)
        input_t = torch.from_numpy(input_combined).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model_dense(input_t)
            dense_probs = F.softmax(output, dim=1)[0].cpu().numpy()

    # SOTA Weighted Ensemble (Consensus)
    # Clinical features are generally more reliable for ACDC, so we give XGBoost more weight
    hybrid_probs = 0.7 * xgb_probs + 0.3 * dense_probs
    pred_idx = np.argmax(hybrid_probs)
    
    class_map_inv = {0: "NOR", 1: "MINF", 2: "DCM", 3: "HCM", 4: "RV"}
    prediction = class_map_inv.get(pred_idx, "UNKNOWN")
    
    print(f"\n--- SOTA Hybrid Diagnosis ---")
    print(f"XGBoost Prediction: {class_map_inv.get(np.argmax(xgb_probs))}")
    print(f"DenseNet Prediction: {class_map_inv.get(np.argmax(dense_probs))}")
    print(f"FINAL CONSENSUS: {prediction} (Prob: {np.max(hybrid_probs):.2f})")
    print(f"TRUE DIAGNOSIS: {patient['diagnosis']}")

    # 4. Grad-CAM Visualization
    if os.path.exists(densenet_path):
        print("\nGenerating SOTA Grad-CAM focus map...")
        target_layer = model_dense.densenet.features.denseblock4.denselayer16
        gcam = GradCAM(model_dense, target_layer)
        heatmap = gcam.generate_heatmap(input_t, pred_idx)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(ed_s, cmap='gray'); plt.title("Original MRI")
        plt.subplot(1, 2, 2); plt.imshow(ed_s, cmap='gray'); plt.imshow(heatmap[0], cmap='jet', alpha=0.5)
        plt.title(f"Grad-CAM: {prediction}")
        plt.savefig(config.GRADCAM_PLOT_PATH)

if __name__ == "__main__":
    run_pipeline(0)
