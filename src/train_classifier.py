import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import joblib

import config
from dataloader import ACDC_PatientDataset
from inference import get_automated_features
from visualize import plot_diagnosis_metrics, plot_feature_tsne, plot_clinical_boxplots

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def prepare_automated_data(dataset, fold, device):
    """
    Iterate over all patients in `dataset`, run the fold-specific UNet to
    generate predicted masks, and extract the 18 clinical features.

    Returns
    -------
    X : numpy array  (N, 18)
    y : numpy array  (N,)  — string diagnosis labels
    """
    X, y = [], []
    print(f"  Extracting AUTOMATED features for fold {fold} "
          f"({len(dataset)} patients) …")
    for i in range(len(dataset)):
        patient = dataset[i]

        feats = get_automated_features(patient, fold, device)
        X.append([feats[k] for k in config.FEATURE_NAMES])
        y.append(patient["diagnosis"])
        if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
            print(f"    {i+1}/{len(dataset)} patients processed …")
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train_xgboost():
    device       = config.DEVICE
    dataset_path = config.DATA_DIR

    all_fold_acc = []
    all_y_true   = []
    all_y_pred   = []
    all_X_train  = []
    all_y_train  = []

    for fold in range(5):
        print(f"\n{'='*45}")
        print(f"   TRAINING CLASSIFIER  —  FOLD {fold}")
        print(f"{'='*45}")

        # ------------------------------------------------------------------ #
        # 1. Extract features from automated (predicted) masks               #
        # ------------------------------------------------------------------ #
        train_dataset = ACDC_PatientDataset(
            root_dir=dataset_path, fold=fold, mode="train"
        )
        X_train, y_train = prepare_automated_data(train_dataset, fold, device)

        val_dataset = ACDC_PatientDataset(
            root_dir=dataset_path, fold=fold, mode="val"
        )
        X_val, y_val = prepare_automated_data(val_dataset, fold, device)

        # ------------------------------------------------------------------ #
        # 2. Label encoding — fit on train, transform both                   #
        # ------------------------------------------------------------------ #
        le          = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_val_enc   = le.transform(y_val)

        # ------------------------------------------------------------------ #
        # 3. XGBoost with RandomisedSearchCV                                 #
        # ------------------------------------------------------------------ #
        xgb = XGBClassifier(
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=config.RANDOM_SEED,
        )
        param_grid = {
            'n_estimators':  [50, 100, 150],
            'max_depth':     [3, 4, 5],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample':     [0.8, 1.0],
        }
        model_search = RandomizedSearchCV(
            xgb, param_distributions=param_grid,
            n_iter=10, cv=3,
            random_state=config.RANDOM_SEED, n_jobs=-1,
        )
        model_search.fit(X_train, y_train_enc)
        model = model_search.best_estimator_
        print(f"  Best params (fold {fold}): {model_search.best_params_}")

        # ------------------------------------------------------------------ #
        # 4. Evaluate                                                         #
        # ------------------------------------------------------------------ #
        y_pred_enc = model.predict(X_val)
        acc        = accuracy_score(y_val_enc, y_pred_enc)
        print(f"\n  Fold {fold} Accuracy: {acc:.4f}")

        print(classification_report(
            y_val_enc, y_pred_enc,
            target_names=le.classes_, zero_division=0,
        ))

        all_fold_acc.append(acc)
        all_y_true.extend(le.inverse_transform(y_val_enc))
        all_y_pred.extend(le.inverse_transform(y_pred_enc))
        all_X_train.append(X_train)
        all_y_train.extend(y_train)

        # ------------------------------------------------------------------ #
        # 5. Save fold model and label encoder                               #
        # ------------------------------------------------------------------ #
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        os.makedirs(config.MODELS_DIR,  exist_ok=True)

        fold_model_path = config.RF_MODEL_PATH.replace(
            ".joblib", f"_fold{fold}.joblib"
        )
        enc_path = os.path.join(
            config.RESULTS_DIR, f"encoder_fold{fold}.joblib"
        )
        joblib.dump(model, fold_model_path)
        joblib.dump(le,    enc_path)
        print(f"  Saved model   : {fold_model_path}")
        print(f"  Saved encoder : {enc_path}")

    # ---------------------------------------------------------------------- #
    # 6. Aggregate metrics and visualisations                                #
    # ---------------------------------------------------------------------- #
    mean_acc = np.mean(all_fold_acc)
    std_acc  = np.std(all_fold_acc)
    print(f"\n{'='*45}")
    print(f"   FINAL 5-FOLD ACCURACY: {mean_acc:.4f} (+/- {std_acc:.4f})")
    print(f"   Per-fold: {[f'{a:.4f}' for a in all_fold_acc]}")
    print(f"{'='*45}")

    # Confusion matrix over all folds combined
    plot_diagnosis_metrics(
        all_y_true, all_y_pred,
        labels=config.DIAGNOSIS_CLASSES,
        save_name="confusion_matrix_all_folds.png",
    )

    # t-SNE of all training features
    X_all = np.vstack(all_X_train)
    plot_feature_tsne(X_all, all_y_train, save_name="feature_tsne.png")

    # Boxplots of clinical feature distributions per class
    plot_clinical_boxplots(
        X_all, all_y_train,
        feature_names=config.FEATURE_NAMES,
        save_name="clinical_distributions.png",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_xgboost()
