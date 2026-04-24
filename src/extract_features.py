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
    
    # Ejection Fraction
    lv_ef = (ed_lv_vol - es_lv_vol) / (ed_lv_vol + 1e-8) * 100
    rv_ef = (ed_rv_vol - es_rv_vol) / (ed_rv_vol + 1e-8) * 100
    
    # Ratios (Extra features for fine-tuning)
    lv_rv_ratio = ed_lv_vol / (ed_rv_vol + 1e-8)
    mass_vol_ratio = (ed_myo_vol * 1.05) / (ed_lv_vol + 1e-8)
    
    features = {
        "LV_EDV": ed_lv_vol,
        "LV_ESV": es_lv_vol,
        "LV_EF": lv_ef,
        "RV_EDV": ed_rv_vol,
        "RV_ESV": es_rv_vol,
        "RV_EF": rv_ef,
        "Myo_Mass": ed_myo_vol * 1.05,
        "LV_RV_Ratio": lv_rv_ratio,
        "Mass_Vol_Ratio": mass_vol_ratio
    }
    
    return features

if __name__ == "__main__":
    # Mock test
    print("Testing feature extraction...")
    mask_ed = np.zeros((10, 256, 256))
    mask_ed[:, 100:150, 100:150] = 3 # LV
    mask_ed[:, 50:100, 50:100] = 1 # RV
    mask_ed[:, 100:110, 100:150] = 2 # Myo
    
    mask_es = np.zeros((10, 256, 256))
    mask_es[:, 110:140, 110:140] = 3 # LV smaller
    mask_es[:, 60:90, 60:90] = 1 # RV smaller
    
    spacing = (1.5, 1.5, 10.0)
    
    feats = extract_patient_features(mask_ed, mask_es, spacing)
    for k, v in feats.items():
        print(f"{k}: {v:.2f}")
