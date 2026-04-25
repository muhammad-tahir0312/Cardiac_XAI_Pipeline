import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import joblib

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        # SOTA Fix: Clone output to avoid in-place modification errors
        self.activations = output.clone()

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple, we take the first element (gradients w.r.t. the output of the layer)
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # If output is (Batch, Classes, H, W), we target the class
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Guided Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        cam = F.relu(cam)
        
        # Upsample to input size
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam

def explain_diagnosis_shap(model, X_train, X_test, feature_names, save_path):
    """
    Generate SHAP summary plot for Random Forest diagnosis
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"SHAP summary plot saved to {save_path}")

if __name__ == "__main__":
    # Test SHAP with the trained model
    model_path = "/home/tahir/Cardiac_XAI_Pipeline/results/diagnosis_classifier.joblib"
    if os.path.exists(model_path):
        print("Testing SHAP Explainer...")
        clf = joblib.load(model_path)
        
        # Load some data to explain
        from train_classifier import prepare_data
        from dataloader import ACDC_PatientDataset
        dataset_path = "/home/tahir/Cardiac_XAI_Pipeline/data"
        val_dataset = ACDC_PatientDataset(root_dir=dataset_path, split="validation_set")
        X_val, y_val, _ = prepare_data(val_dataset)
        
        feature_names = ["LV_EDV", "LV_ESV", "LV_EF", "RV_EDV", "RV_ESV", "RV_EF", "Myo_Mass"]
        save_path = "/home/tahir/Cardiac_XAI_Pipeline/results/shap_summary.png"
        explain_diagnosis_shap(clf, X_val, X_val, feature_names, save_path)
    else:
        print("Model not found. Run train_classifier.py first.")
