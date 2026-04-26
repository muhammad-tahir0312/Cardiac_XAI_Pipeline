import torch
import torch.nn.functional as F
import numpy as np
import os
from networks import UNet
import config
from scipy.ndimage import label as scipy_label

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def post_process_mask(mask):
    """
    Keep only the largest connected component per foreground class.
    Removes small false-positive fragments for clinical consistency.
    Labels: 1 = RV cavity, 2 = Myocardium, 3 = LV cavity.
    """
    refined_mask = np.zeros_like(mask)
    for c in range(1, 4):
        class_mask = (mask == c)
        labeled_array, num_features = scipy_label(class_mask)
        if num_features > 0:
            largest_cc, max_size = 0, 0
            for i in range(1, num_features + 1):
                size = np.sum(labeled_array == i)
                if size > max_size:
                    max_size = size
                    largest_cc = i
            refined_mask[labeled_array == largest_cc] = c
    return refined_mask


# ---------------------------------------------------------------------------
# Single-slice inference helper
# ---------------------------------------------------------------------------

def _predict_single_slice(model, slice_2d, device):
    """
    Normalise, pad/crop one 2-D slice to 256×256, run the model, and
    return the softmax probability map at 256×256 resolution.

    The caller is responsible for inverting the spatial transform.

    Returns
    -------
    prob   : torch.Tensor  (1, 4, 256, 256) on CPU
    offsets: dict with keys pad_top, pad_left, crop_top, crop_left,
             h_orig, w_orig — everything needed to invert the transform.
    """
    h_orig, w_orig = slice_2d.shape

    # Z-score normalisation
    norm = (slice_2d - np.mean(slice_2d)) / (np.std(slice_2d) + 1e-8)

    # --- Pad to at least 256×256, recording EXACT offsets ---
    ph_total   = max(0, 256 - h_orig)
    pw_total   = max(0, 256 - w_orig)
    pad_top    = ph_total // 2
    pad_bottom = ph_total - pad_top
    pad_left   = pw_total // 2
    pad_right  = pw_total - pad_left

    padded = np.pad(norm,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant')

    # --- Crop to exactly 256×256 if the padded image is still larger ---
    h_pad, w_pad = padded.shape
    crop_top  = (h_pad - 256) // 2
    crop_left = (w_pad - 256) // 2
    inp = padded[crop_top: crop_top + 256, crop_left: crop_left + 256]

    input_t = (torch.from_numpy(inp.astype(np.float32))
               .unsqueeze(0).unsqueeze(0).to(device))

    with torch.no_grad():
        logits = model(input_t)          # (1, 4, 256, 256)
        prob   = torch.softmax(logits, dim=1).cpu()

    offsets = dict(
        pad_top=pad_top, pad_left=pad_left,
        crop_top=crop_top, crop_left=crop_left,
        h_orig=h_orig, w_orig=w_orig,
    )
    return prob, offsets


def _invert_transform(pred_256, offsets):
    """
    Invert the pad→crop transform applied in _predict_single_slice to
    recover a prediction at the original (h_orig, w_orig) resolution.

    For images smaller than 256: we padded first, so the valid region of
    the 256×256 output starts at (pad_top, pad_left) and has size
    min(h_orig, 256) × min(w_orig, 256).

    For images larger than 256: we also cropped, so we must account for
    the crop offset when placing the prediction back.
    """
    h_orig    = offsets["h_orig"]
    w_orig    = offsets["w_orig"]
    pad_top   = offsets["pad_top"]
    pad_left  = offsets["pad_left"]
    crop_top  = offsets["crop_top"]
    crop_left = offsets["crop_left"]

    pred_final = np.zeros((h_orig, w_orig), dtype=pred_256.dtype)

    # Region inside 256×256 that corresponds to the (padded) original
    # After padding: valid rows are [pad_top : pad_top + h_orig] inside padded
    # After cropping: subtract crop_top to get coords inside the 256×256 pred
    row_start = pad_top  - crop_top
    col_start = pad_left - crop_left

    # How many rows/cols of the original fit inside the 256×256 prediction
    row_end_pred = row_start + h_orig   # may extend beyond 256
    col_end_pred = col_start + w_orig

    # Clamp to valid [0, 256] range
    r0p = max(row_start, 0);  r1p = min(row_end_pred, 256)
    c0p = max(col_start, 0);  c1p = min(col_end_pred, 256)

    # Corresponding coordinates in pred_final
    r0f = r0p - row_start;  r1f = r0f + (r1p - r0p)
    c0f = c0p - col_start;  c1f = c0f + (c1p - c0p)

    pred_final[r0f:r1f, c0f:c1f] = pred_256[r0p:r1p, c0p:c1p]
    return pred_final


# ---------------------------------------------------------------------------
# Full-volume inference with TTA
# ---------------------------------------------------------------------------

def predict_full_volume(model, volume, device):
    """
    Predict segmentation masks for a full 3-D volume (Slices, H, W).

    Test-Time Augmentation (TTA): each slice is predicted three times —
    original, horizontal flip, vertical flip — probability maps are averaged
    before argmax.  This matches the paper's claimed TTA strategy and
    improves boundary accuracy.

    Parameters
    ----------
    model  : trained UNet
    volume : numpy array (S, H, W)
    device : torch device

    Returns
    -------
    refined_mask : numpy array (S, H_orig, W_orig) dtype uint8
    """
    model.eval()
    predicted_mask = []

    for s in range(volume.shape[0]):
        raw_slice = volume[s].astype(np.float32)

        # ---- TTA: original + h-flip + v-flip ----
        prob_orig,  offsets = _predict_single_slice(model, raw_slice,             device)
        prob_hflip, _       = _predict_single_slice(model, np.fliplr(raw_slice),  device)
        prob_vflip, _       = _predict_single_slice(model, np.flipud(raw_slice),  device)

        # Un-flip the augmented probability maps before averaging
        prob_hflip = torch.flip(prob_hflip, dims=[3])   # undo left-right flip
        prob_vflip = torch.flip(prob_vflip, dims=[2])   # undo up-down flip

        avg_prob = (prob_orig + prob_hflip + prob_vflip) / 3.0
        pred_256 = torch.argmax(avg_prob, dim=1)[0].numpy()   # (256, 256)

        # ---- Invert pad/crop to restore original spatial size ----
        pred_orig = _invert_transform(pred_256, offsets)
        predicted_mask.append(pred_orig)

    full_mask = np.stack(predicted_mask, axis=0)

    # Largest-connected-component post-processing per class
    refined_mask = np.zeros_like(full_mask)
    for s in range(full_mask.shape[0]):
        refined_mask[s] = post_process_mask(full_mask[s])

    return refined_mask


# ---------------------------------------------------------------------------
# Zero-leakage automated feature extraction (used by train_classifier + predict)
# ---------------------------------------------------------------------------

def get_automated_features(patient_data, fold, device):
    """
    Zero-Leakage automated clinical feature extraction.

    Loads the fold-specific UNet, generates predicted segmentation masks
    from raw MRI, then computes clinical metrics from those predicted masks
    (never from ground-truth annotations).

    Parameters
    ----------
    patient_data : dict returned by ACDC_PatientDataset.__getitem__
    fold         : int — which fold's UNet weights to use
    device       : torch device

    Returns
    -------
    dict of 18 clinical feature values
    """
    from extract_features import extract_patient_features

    model_path = config.SEG_MODEL_PATH.replace(".pth", f"_fold{fold}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Segmentation model for fold {fold} not found at '{model_path}'.\n"
            "Run train_seg.py first."
        )

    model = UNet(in_chns=1, class_num=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    ed_mask_pred = predict_full_volume(model, patient_data["ed_image"], device)
    es_mask_pred = predict_full_volume(model, patient_data["es_image"], device)

    features = extract_patient_features(
        ed_mask_pred,
        es_mask_pred,
        patient_data["spacing"],
        bsa=patient_data["bsa"],
    )
    return features
