import numpy as np

def calculate_volume(mask, label, spacing):
    """
    Calculate volume for a specific label in the mask.
    Args:
        mask    : 3-D numpy array (Slices, Height, Width)
        label   : int  — 1: RV cavity, 2: Myocardium, 3: LV cavity
        spacing : tuple (dx, dy, dz) in mm

    Returns:
        float: Volume in mm³
    """
    num_pixels   = np.sum(mask == label)
    pixel_volume = spacing[0] * spacing[1] * spacing[2]
    return float(num_pixels * pixel_volume)


def extract_patient_features(ed_mask, es_mask, spacing, bsa=1.7):
    """
    Extract 18 clinical features from ED and ES segmentation masks.

    All volumes are in ml (mm³ / 1000).
    Indexed volumes are normalised by Body Surface Area (BSA) in m².

    Parameters
    ----------
    ed_mask  : 3-D numpy array (Slices, H, W) — End-Diastolic segmentation
    es_mask  : 3-D numpy array (Slices, H, W) — End-Systolic segmentation
    spacing  : (dx, dy, dz) voxel spacing in mm
    bsa      : Body Surface Area in m²  (default 1.7 = population mean)

    Returns
    -------
    dict mapping feature name → float value
    """
    # ------------------------------------------------------------------ #
    # Raw volumes in ml (divide mm³ by 1000)                             #
    # ------------------------------------------------------------------ #
    ed_lv_vol = calculate_volume(ed_mask, 3, spacing) / 1000.0   # ml  LV at ED
    es_lv_vol = calculate_volume(es_mask, 3, spacing) / 1000.0   # ml  LV at ES

    ed_rv_vol = calculate_volume(ed_mask, 1, spacing) / 1000.0   # ml  RV at ED
    es_rv_vol = calculate_volume(es_mask, 1, spacing) / 1000.0   # ml  RV at ES

    ed_myo_vol_ml = calculate_volume(ed_mask, 2, spacing) / 1000.0  # ml
    es_myo_vol_ml = calculate_volume(es_mask, 2, spacing) / 1000.0  # ml

    # ------------------------------------------------------------------ #
    # Derived cardiac function metrics                                    #
    # ------------------------------------------------------------------ #
    lv_sv = ed_lv_vol - es_lv_vol          # Stroke Volume  (ml)
    rv_sv = ed_rv_vol - es_rv_vol          # RV Stroke Volume  (ml)

    lv_ef = lv_sv / (ed_lv_vol + 1e-8) * 100   # LV Ejection Fraction (%)
    rv_ef = rv_sv / (ed_rv_vol + 1e-8) * 100   # RV Ejection Fraction (%)

    # ------------------------------------------------------------------ #
    # Myocardial thickening index (important for MINF detection)         #
    # Positive = wall thickens during systole (normal)                   #
    # Negative = wall thins (indicative of infarct)                      #
    # ------------------------------------------------------------------ #
    myo_thickening = (
        (es_myo_vol_ml - ed_myo_vol_ml) / (ed_myo_vol_ml + 1e-8) * 100
    )  # %

    # RV Fractional Area Change surrogate (via volumes)
    rv_fac = (ed_rv_vol - es_rv_vol) / (ed_rv_vol + 1e-8) * 100  # %

    # ------------------------------------------------------------------ #
    # Morphological / shape ratios                                        #
    # ------------------------------------------------------------------ #
    lv_rv_ratio = ed_lv_vol / (ed_rv_vol + 1e-8)

    # Myocardial mass in grams (myocardial tissue density ≈ 1.05 g/ml)
    myo_mass_g = ed_myo_vol_ml * 1.05   # g

    # Mass-to-volume ratio: elevated in HCM (concentric hypertrophy)
    # Units: g/ml
    mass_vol_ratio = myo_mass_g / (ed_lv_vol + 1e-8)

    # computed as a geometric proxy using cube-root of LV_EDV as a
    # surrogate for LVEDD — gives independent discriminative information.
    # True RWT = 2 × posterior wall thickness / LVEDD requires contours.
    # Units: g/ml^(2/3)  — a shape-normalised mass index
    rwt = myo_mass_g / (ed_lv_vol ** (1.0 / 3.0) + 1e-8)

    # ------------------------------------------------------------------ #
    # BSA-indexed volumes (standard clinical reporting)                   #
    # ------------------------------------------------------------------ #
    ed_lv_index = ed_lv_vol  / bsa   # ml/m²   LVEDVi
    es_lv_index = es_lv_vol  / bsa   # ml/m²   LVESVi
    sv_index    = lv_sv      / bsa   # ml/m²   SVi
    mass_index  = myo_mass_g / bsa   # g/m²    Massi

    features = {
        "LV_EDV":         ed_lv_vol,
        "LV_ESV":         es_lv_vol,
        "LV_SV":          lv_sv,
        "LV_EF":          lv_ef,
        "RV_EDV":         ed_rv_vol,
        "RV_ESV":         es_rv_vol,
        "RV_SV":          rv_sv,
        "RV_EF":          rv_ef,
        "Myo_Mass":       myo_mass_g,
        "LV_RV_Ratio":    lv_rv_ratio,
        "Mass_Vol_Ratio": mass_vol_ratio,
        "RWT":            rwt,
        "LVEDVi":         ed_lv_index,
        "LVESVi":         es_lv_index,
        "SVi":            sv_index,
        "Massi":          mass_index,
        "Myo_Thickening": myo_thickening,
        "RV_FAC":         rv_fac,
    }

    return features


if __name__ == "__main__":
    # Sanity check with synthetic data
    rng = np.random.default_rng(0)
    dummy_ed = np.zeros((10, 128, 128), dtype=np.uint8)
    dummy_ed[3:7, 40:90, 40:90] = 3    # LV cavity
    dummy_ed[3:7, 35:95, 35:95] = 2    # Myocardium ring (overwritten by LV)
    dummy_ed[3:7, 60:100, 10:50] = 1   # RV cavity

    dummy_es = dummy_ed.copy()
    dummy_es[4:6, 45:85, 45:85] = 3    # smaller LV at ES

    spacing = (1.5, 1.5, 8.0)          # typical ACDC spacing in mm
    feats = extract_patient_features(dummy_ed, dummy_es, spacing, bsa=1.8)

    print("Feature extraction sanity check:")
    for k, v in feats.items():
        print(f"  {k:20s}: {v:.4f}")

    # Verify RWT and Mass_Vol_Ratio are no longer identical
    assert feats["RWT"] != feats["Mass_Vol_Ratio"], \
        "RWT must differ from Mass_Vol_Ratio!"
    print("\nAll assertions passed.")
