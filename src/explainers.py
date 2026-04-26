import os
import joblib
import torch
import torch.nn.functional as F
import numpy as np
import shap
import matplotlib.pyplot as plt

import config

# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM implementation for DenseNetDiagnosis.

    Registers forward and backward hooks on a target convolutional layer,
    weights the activation maps by the gradient signal, and returns a
    normalised heatmap at the input spatial resolution.

    Usage
    -----
    gcam    = GradCAM(model, target_layer)
    heatmap = gcam.generate_heatmap(input_tensor, class_idx)
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.clone()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; element 0 is the gradient w.r.t. layer output
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        """
        Generate a Grad-CAM heatmap for the given input and target class.

        Parameters
        ----------
        input_tensor : torch.Tensor  (1, C, H, W) on the model's device
        class_idx    : int — index of the target class

        Returns
        -------
        heatmap : numpy array (H, W), values in [0, 1]
        """
        input_tensor = input_tensor.clone().requires_grad_(True)
        self.model.zero_grad()

        output = self.model(input_tensor)

        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w))

        # Global average pool the gradients → per-channel weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam     = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam     = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False,
        )

        cam = cam.squeeze().cpu().detach().numpy()
        # Normalise to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ---------------------------------------------------------------------------
# SHAP — summary plot (all patients at once)
# ---------------------------------------------------------------------------

def explain_diagnosis_shap(model, X_train, X_test, feature_names, save_path):
    """
    Generate a SHAP beeswarm summary plot for the XGBoost clinical classifier.

    Parameters
    ----------
    model        : trained XGBoost estimator
    X_train      : background dataset (numpy array) for the SHAP explainer
    X_test       : samples to explain (numpy array)
    feature_names: list of feature name strings
    save_path    : file path for the saved PNG
    """
    explainer   = shap.TreeExplainer(model, data=X_train)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names, show=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"SHAP summary plot saved to {save_path}")


# ---------------------------------------------------------------------------
# SHAP — single-patient waterfall / bar chart (used by predict.py)
# ---------------------------------------------------------------------------

def extract_shap_values_for_class(shap_values, class_idx):
    """
    Robustly extract per-feature SHAP values for one sample and one class
    from any of the three output formats produced by different SHAP / XGBoost
    version combinations.

    Formats handled
    ---------------
    (a) list of arrays, each (n_samples, n_features)  — one per class
    (b) 3-D array (n_samples, n_features, n_classes)  — newer SHAP ≥ 0.40
    (c) 2-D array (n_samples, n_features)              — single-output fallback

    Parameters
    ----------
    shap_values : raw output of explainer.shap_values(X)
    class_idx   : int — which class to pull

    Returns
    -------
    sv : 1-D numpy array (n_features,) for sample 0
    """
    if isinstance(shap_values, list):
        # Format (a): list[class_idx] → (n_samples, n_features) → row 0
        print(f"[SHAP] Format: list of {len(shap_values)} arrays, "
              f"each shape {shap_values[0].shape}")
        sv = shap_values[class_idx][0]

    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Format (b): (n_samples, n_features, n_classes)
        print(f"[SHAP] Format: 3-D array {shap_values.shape}")
        sv = shap_values[0, :, class_idx]

    else:
        # Format (c): (n_samples, n_features)
        print(f"[SHAP] Format: 2-D array {np.array(shap_values).shape}")
        sv = np.array(shap_values)[0]

    return sv


# ---------------------------------------------------------------------------
# Smoke test / standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    from dataloader import ACDC_PatientDataset
    from train_classifier import prepare_automated_data

    clf_path = config.RF_MODEL_PATH.replace(".joblib", "_fold0.joblib")

    if not os.path.exists(clf_path):
        print(f"Model not found at {clf_path}. Run train_classifier.py first.")
    else:
        print("Testing SHAP Explainer (fold 0)…")
        clf = joblib.load(clf_path)

        dataset_path = config.DATA_DIR
        
        val_dataset = ACDC_PatientDataset(
            root_dir=dataset_path, fold=0, mode="val"
        )
        X_val, y_val = prepare_automated_data(
            val_dataset, fold=0, device=config.DEVICE
        )

        explain_diagnosis_shap(
            clf,
            X_train=X_val,
            X_test=X_val,
            feature_names=config.FEATURE_NAMES,
            save_path=config.SHAP_PLOT_PATH,
        )
