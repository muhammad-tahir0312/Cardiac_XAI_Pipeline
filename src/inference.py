import torch
import torch.nn.functional as F
import numpy as np
import os
from networks import UNet
import config
from scipy.ndimage import label
from tqdm import tqdm

def post_process_mask(mask):
    """
    Keep only the largest connected component for each foreground class.
    This ensures clinical consistency by removing small false positive artifacts.
    """
    refined_mask = np.zeros_like(mask)
    for c in range(1, 4): # Labels 1, 2, 3 (RV, Myo, LV)
        class_mask = (mask == c)
        labeled_array, num_features = label(class_mask)
        if num_features > 0:
            largest_cc = 0
            max_size = 0
            for i in range(1, num_features + 1):
                size = np.sum(labeled_array == i)
                if size > max_size:
                    max_size = size
                    largest_cc = i
            refined_mask[labeled_array == largest_cc] = c
    return refined_mask

def predict_full_volume(model, volume, device):
    """
    Predict masks for a full 3D volume slice-by-slice.
    Args:
        model: Trained UNet model
        volume: Numpy array (Slices, H, W)
        device: torch device
    Returns:
        Predicted 3D mask (Slices, H, W)
    """
    model.eval()
    predicted_mask = []
    
    # Process each slice with normalization and padding
    for s in range(volume.shape[0]):
        mri_slice = volume[s].astype(np.float32)
        
        # Z-score Normalization (SOTA Practice)
        mri_slice_norm = (mri_slice - np.mean(mri_slice)) / (np.std(mri_slice) + 1e-8)
        
        # Standardize input to 256x256
        h, w = mri_slice_norm.shape
        ph, pw = max(0, 256 - h), max(0, 256 - w)
        mri_slice_norm = np.pad(mri_slice_norm, ((ph//2, ph-ph//2), (pw//2, pw-pw//2)), mode='constant')
        sh, sw = (mri_slice_norm.shape[0] - 256) // 2, (mri_slice_norm.shape[1] - 256) // 2
        mri_slice_norm = mri_slice_norm[sh:sh+256, sw:sw+256]
        
        input_t = torch.from_numpy(mri_slice_norm).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_t)
            pred = torch.argmax(output, dim=1)[0].cpu().numpy()
        
        # Restore original dimensions
        rh, rw = h, w
        rph, rpw = max(0, 256 - rh), max(0, 256 - rw)
        rsh, rsw = rph // 2, rpw // 2
        pred_cropped = pred[rsh:rsh+rh, rsw:rsw+rw]
        
        predicted_mask.append(pred_cropped)
        
    full_mask = np.stack(predicted_mask, axis=0)
    
    # Final morphological refinement
    refined_mask = np.zeros_like(full_mask)
    for s in range(full_mask.shape[0]):
        refined_mask[s] = post_process_mask(full_mask[s])
        
    return refined_mask

def get_automated_features(patient_data, fold, device):
    """
    SOTA Automated Clinical Feature Extraction (Zero Leakage).
    Derives features from automated segmentation instead of ground truth.
    """
    from extract_features import extract_patient_features
    
    # Load the specific fold model
    model_path = config.SEG_MODEL_PATH.replace(".pth", f"_fold{fold}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for fold {fold} not found at {model_path}.")
    
    model = UNet(in_chns=1, class_num=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Generate volume masks
    ed_mask_pred = predict_full_volume(model, patient_data["ed_image"], device)
    es_mask_pred = predict_full_volume(model, patient_data["es_image"], device)
    
    # Calculate clinical metrics
    features = extract_patient_features(ed_mask_pred, es_mask_pred, patient_data["spacing"], bsa=patient_data["bsa"])
    return features
