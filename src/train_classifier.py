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

from dataloader import ACDC_PatientDataset
from inference import get_automated_features
from visualize import plot_diagnosis_metrics, plot_feature_tsne, plot_clinical_boxplots
import config

def prepare_automated_data(dataset, fold, device):
    X = []
    y = []
    print(f"Extracting AUTOMATED features for fold {fold} ({len(dataset)} patients)...")
    for i in range(len(dataset)):
        patient = dataset[i]
        # Use predicted masks, not GT!
        feats = get_automated_features(patient, fold, device)
        X.append([feats[k] for k in config.FEATURE_NAMES])
        y.append(patient["diagnosis"])
    return np.array(X), np.array(y)

def train_xgboost():
    device = config.DEVICE
    dataset_path = config.DATA_DIR
    
    all_fold_acc = []
    
    # Run 5-Fold Cross-Validation
    for fold in range(5):
        print(f"\n" + "="*40)
        print(f"   TRAINING CLASSIFIER FOLD {fold}")
        print(f"="*40)
        
        # 1. Prepare Automated Data (No Leakage)
        train_dataset = ACDC_PatientDataset(root_dir=dataset_path, fold=fold, mode="train")
        X_train, y_train = prepare_automated_data(train_dataset, fold, device)
        
        val_dataset = ACDC_PatientDataset(root_dir=dataset_path, fold=fold, mode="val")
        X_val, y_val = prepare_automated_data(val_dataset, fold, device)
        
        # 2. Encode Labels
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(y_val)
        
        # 3. Train
        xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1, 0.2]
        }
        model_search = RandomizedSearchCV(xgb, param_distributions=param_grid, n_iter=10, cv=3)
        model_search.fit(X_train, y_train_enc)
        model = model_search.best_estimator_
        
        # 4. Evaluate
        y_pred_enc = model.predict(X_val)
        acc = accuracy_score(y_val_enc, y_pred_enc)
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        all_fold_acc.append(acc)
        
        # 5. Save fold models
        fold_model_path = config.RF_MODEL_PATH.replace(".joblib", f"_fold{fold}.joblib")
        joblib.dump(model, fold_model_path)
        joblib.dump(le, os.path.join(config.RESULTS_DIR, f"encoder_fold{fold}.joblib"))

    print(f"\n" + "="*40)
    print(f"   FINAL 5-FOLD ACCURACY: {np.mean(all_fold_acc):.4f} (+/- {np.std(all_fold_acc):.4f})")
    print(f"="*40)

if __name__ == "__main__":
    train_xgboost()
