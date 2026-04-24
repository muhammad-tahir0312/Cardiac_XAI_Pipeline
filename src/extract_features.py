import numpy as np

def calculate_volume(mask, label, spacing):
    """
    Calculate volume for a specific label in the mask.
    Args:
        mask (numpy array): 3D mask (Slices, Height, Width)
        label (int): Label to calculate volume for (1: RV, 2: Myo, 3: LV)
        spacing (tuple): (dx, dy, dz)
    Returns:
        float: Volume in mm^3 (can be converted to ml by dividing by 1000)
    """
    num_pixels = np.sum(mask == label)
    pixel_volume = spacing[0] * spacing[1] * spacing[2]
    return num_pixels * pixel_volume

def extract_patient_features(ed_mask, es_mask, spacing):
    """
    Extract clinical features from ED and ES masks.
    """
    # Volumes in ml
    ed_lv_vol = calculate_volume(ed_mask, 3, spacing) / 1000.0
    es_lv_vol = calculate_volume(es_mask, 3, spacing) / 1000.0
    
    ed_rv_vol = calculate_volume(ed_mask, 1, spacing) / 1000.0
    es_rv_vol = calculate_volume(es_mask, 1, spacing) / 1000.0
    
    ed_myo_vol = calculate_volume(ed_mask, 2, spacing) / 1000.0
    
    # Stroke Volumes
    lv_sv = ed_lv_vol - es_lv_vol
    rv_sv = ed_rv_vol - es_rv_vol
    
    # Ejection Fraction
    lv_ef = lv_sv / (ed_lv_vol + 1e-8) * 100
    rv_ef = rv_sv / (ed_rv_vol + 1e-8) * 100
    
    # Ratios (Extra features for fine-tuning)
    lv_rv_ratio = ed_lv_vol / (ed_rv_vol + 1e-8)
    mass_vol_ratio = (ed_myo_vol * 1.05) / (ed_lv_vol + 1e-8)
    
    features = {
        "LV_EDV": ed_lv_vol,
        "LV_ESV": es_lv_vol,
        "LV_SV": lv_sv,
        "LV_EF": lv_ef,
        "RV_EDV": ed_rv_vol,
        "RV_ESV": es_rv_vol,
        "RV_SV": rv_sv,
        "RV_EF": rv_ef,
        "Myo_Mass": ed_myo_vol * 1.05,
        "LV_RV_Ratio": lv_rv_ratio,
        "Mass_Vol_Ratio": mass_vol_ratio
    }
    
    return features

if __name__ == "__main__":
    # This module is intended to be imported.
    # Use train_classifier.py to extract features for the full dataset.
    pass
