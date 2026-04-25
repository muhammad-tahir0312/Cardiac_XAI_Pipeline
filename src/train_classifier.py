import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import joblib

from dataloader import ACDC_PatientDataset
from extract_features import extract_patient_features
from visualize import plot_diagnosis_metrics, plot_feature_tsne, plot_clinical_boxplots
import config

def prepare_data(dataset):
    X = []
    y = []
    ids = []
    print(f"Extracting features for {len(dataset)} patients...")
    for i in range(len(dataset)):
        patient = dataset[i]
        feats = extract_patient_features(patient["ed_mask"], patient["es_mask"], patient["spacing"], bsa=patient["bsa"])
        X.append([
            feats["LV_EDV"], feats["LV_ESV"], feats["LV_SV"], feats["LV_EF"],
            feats["RV_EDV"], feats["RV_ESV"], feats["RV_SV"], feats["RV_EF"],
            feats["Myo_Mass"], feats["LV_RV_Ratio"], feats["Mass_Vol_Ratio"],
            feats["RWT"], feats["LVEDVi"], feats["LVESVi"], feats["SVi"], feats["Massi"],
            feats["Myo_Thickening"], feats["RV_FAC"]
        ])
        y.append(patient["diagnosis"])
        ids.append(patient["patient_id"])
    return np.array(X), np.array(y), ids

def train_xgboost():
    dataset_path = config.DATA_DIR
    
    # 1. Prepare Data
    train_dataset = ACDC_PatientDataset(root_dir=dataset_path, split="train_set")
    X_train, y_train, _ = prepare_data(train_dataset)
    
    val_dataset = ACDC_PatientDataset(root_dir=dataset_path, split="validation_set")
    X_val, y_val, _ = prepare_data(val_dataset)
    
    # 2. Encode Labels for XGBoost
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    
    # 3. Train XGBoost with Hyperparameter Tuning
    print("\nTraining XGBoost Classifier with RandomizedSearchCV...")
    xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2]
    }
    
    model_search = RandomizedSearchCV(
        xgb, param_distributions=param_grid, n_iter=50, 
        cv=5, scoring='accuracy', n_jobs=-1, random_state=42
    )
    model_search.fit(X_train, y_train_enc)
    model = model_search.best_estimator_
    print(f"Best Parameters: {model_search.best_params_}")
    
    # 4. Evaluate
    y_pred_enc = model.predict(X_val)
    y_pred = le.inverse_transform(y_pred_enc)
    
    acc = accuracy_score(y_val, y_pred)
    print(f"\nXGBoost Validation Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # 5. Visualizations
    labels = sorted(list(set(y_train)))
    plot_diagnosis_metrics(y_val, y_pred, labels=labels)
    plot_feature_tsne(X_train, y_train)
    plot_clinical_boxplots(X_train, y_train, config.FEATURE_NAMES)
    
    # 6. Save Model and Encoder
    model_path = config.RF_MODEL_PATH
    encoder_path = os.path.join(config.RESULTS_DIR, "label_encoder.joblib")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    
    print(f"XGBoost model saved to {model_path}")
    
    # Also overwrite the main classifier for the predict.py script
    main_model_path = config.RF_MODEL_PATH
    joblib.dump(model, main_model_path)

if __name__ == "__main__":
    train_xgboost()
