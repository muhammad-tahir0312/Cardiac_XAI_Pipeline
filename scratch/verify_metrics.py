
import os
import sys
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add src to path
sys.path.append("/home/tahir/Cardiac_XAI_Pipeline/src")

import config
from dataloader import ACDC_PatientDataset
from extract_features import extract_patient_features

def verify_metrics():
    # Load model and encoder
    model_path = config.RF_MODEL_PATH
    encoder_path = os.path.join(config.RESULTS_DIR, "label_encoder.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print(f"Error: Model or encoder not found.")
        return

    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    
    # Prepare validation data
    dataset_path = config.DATA_DIR
    val_dataset = ACDC_PatientDataset(root_dir=dataset_path, split="validation_set")
    
    X_val = []
    y_val = []
    
    print(f"Extracting features for {len(val_dataset)} patients...")
    for i in range(len(val_dataset)):
        patient = val_dataset[i]
        feats = extract_patient_features(patient["ed_mask"], patient["es_mask"], patient["spacing"])
        X_val.append([
            feats["LV_EDV"], feats["LV_ESV"], feats["LV_SV"], feats["LV_EF"],
            feats["RV_EDV"], feats["RV_ESV"], feats["RV_SV"], feats["RV_EF"],
            feats["Myo_Mass"], feats["LV_RV_Ratio"], feats["Mass_Vol_Ratio"]
        ])
        y_val.append(patient["diagnosis"])
    
    X_val = np.array(X_val)
    y_val_enc = le.transform(y_val)
    
    # Predict
    y_pred_enc = model.predict(X_val)
    y_pred = le.inverse_transform(y_pred_enc)
    
    # Calculate metrics
    acc = accuracy_score(y_val, y_pred)
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    
    print(f"\n--- VERIFICATION RESULTS ---")
    print(f"Validation Accuracy: {acc*100:.1f}%")
    
    report = classification_report(y_val, y_pred, output_dict=True)
    
    # Check DCM and HCM F1-scores specifically if they exist
    for label in ['DCM', 'HCM']:
        if label in report:
            print(f"F1-Score ({label}): {report[label]['f1-score']:.2f}")
    
    print("\nFull Report:")
    print(classification_report(y_val, y_pred))

if __name__ == "__main__":
    verify_metrics()
