import torch
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from dataloader import ACDC_PatientDataset
from networks import UNet
from extract_features import extract_patient_features
from explainers import GradCAM, explain_diagnosis_shap
from visualize import visualize_segmentation, visualize_feature_maps
import config

def run_pipeline(patient_idx=0):
    device = config.DEVICE
    results_dir = config.RESULTS_DIR
    
    # 1. Load Data
    dataset = ACDC_PatientDataset(root_dir=config.DATA_DIR, split="validation_set")
    patient = dataset[patient_idx]
    p_id = patient["patient_id"]
    print(f"Running pipeline for {p_id}...")
    
    # 2. Segmentation (Model Internals & Prediction)
    model_seg = UNet(in_chns=1, class_num=4).to(device)
    if os.path.exists(config.SEG_MODEL_PATH):
        model_seg.load_state_dict(torch.load(config.SEG_MODEL_PATH, map_location=device))
        print("Segmentation model loaded.")
        
        # Visualize Feature Maps (Internals)
        # Using a slice from ED frame
        mid_slice_idx = len(patient["ed_mask"]) // 2
        slice_np = patient["ed_image"][mid_slice_idx].astype(np.float32)
        slice_np = (slice_np - np.min(slice_np)) / (np.max(slice_np) - np.min(slice_np) + 1e-8)
        input_slice = torch.from_numpy(slice_np).to(device)
        visualize_feature_maps(model_seg, input_slice)
    else:
        print("Segmentation weights not found. Using GT for demo.")
    
    # For demonstration, we use GT as "perfect" segmentation
    ed_mask = patient["ed_mask"]
    es_mask = patient["es_mask"]
    
    # Visualization
    visualize_segmentation(patient["ed_mask"][len(patient["ed_mask"])//2], 
                           patient["ed_mask"][len(patient["ed_mask"])//2], 
                           patient["ed_mask"][len(patient["ed_mask"])//2],
                           save_name=f"seg_viz_{p_id}.png")
    
    # 3. Feature Extraction
    features = extract_patient_features(ed_mask, es_mask, patient["spacing"])
    print("\nExtracted Features:")
    for k, v in features.items():
        print(f"{k}: {v:.2f}")
    
    # 4. Diagnosis (Ensemble)
    clf_path = config.RF_MODEL_PATH
    if os.path.exists(clf_path):
        clf = joblib.load(clf_path)
        feature_vector = np.array([[
            features["LV_EDV"], features["LV_ESV"], features["LV_EF"],
            features["RV_EDV"], features["RV_ESV"], features["RV_EF"],
            features["Myo_Mass"], features["LV_RV_Ratio"], features["Mass_Vol_Ratio"]
        ]])
        prediction_idx = clf.predict(feature_vector)[0]
        # Decode label
        encoder_path = os.path.join(results_dir, "label_encoder.joblib")
        if os.path.exists(encoder_path):
            le = joblib.load(encoder_path)
            prediction = le.inverse_transform([prediction_idx])[0]
        else:
            prediction = prediction_idx
            
        print(f"\n[Ensemble Model] Prediction: {prediction}")
        print(f"True Diagnosis: {patient['diagnosis']}")
        
        # SHAP plot is already saved in results/shap_summary.png
    
    # 5. Diagnosis (Deep Learning - DenseNet)
    densenet_path = config.DENSE_MODEL_PATH
    if os.path.exists(densenet_path):
        from networks import DenseNetDiagnosis
        model = DenseNetDiagnosis(class_num=5, pretrained=False).to(device)
        model.load_state_dict(torch.load(densenet_path, map_location=device))
        model.eval()
        
        # Prep input (single slice for demo)
        input_slice = patient["ed_mask"][len(patient["ed_mask"])//2] # middle slice
        input_tensor = torch.from_numpy(input_slice).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            _, pred_idx = torch.max(output, 1)
            class_map_inv = {0: "NOR", 1: "MINF", 2: "DCM", 3: "HCM", 4: "ARV"}
            print(f"[DenseNet Model] Prediction: {class_map_inv.get(pred_idx.item(), 'UNKNOWN')}")
            
        # 6. Grad-CAM
        print("\nGenerating Grad-CAM visualization...")
        # Target the last conv layer of DenseNet
        target_layer = model.densenet.features.norm5 
        gcam = GradCAM(model, target_layer)
        heatmap = gcam.generate_heatmap(input_tensor, pred_idx.item())
        
        # Save visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(input_slice, cmap='gray')
        plt.title("Original MRI Slice")
        plt.subplot(1, 2, 2)
        plt.imshow(input_slice, cmap='gray')
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title("Grad-CAM Focus")
        plt.savefig(config.GRADCAM_PLOT_PATH)
        print(f"Grad-CAM result saved to {config.GRADCAM_PLOT_PATH}")
    else:
        print("\nDenseNet model not found. Run train_densenet.py first.")

if __name__ == "__main__":
    run_pipeline(0)
