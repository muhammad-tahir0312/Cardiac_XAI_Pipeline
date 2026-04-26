import os
import glob
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

import config

# Suppress noisy NIfTI scale warnings from SimpleITK
sitk.ProcessObject.SetGlobalWarningDisplay(False)

# ---------------------------------------------------------------------------
# Shared patient-splitting logic (used by all three dataset classes)
# ---------------------------------------------------------------------------

def get_patient_splits(root_dir, n_splits=5, seed=42):
    """
    Pool train_set + validation_set directories into one list of patient paths,
    then produce a stratified 5-fold split that is identical across all callers.

    Returns
    -------
    splits       : list of (train_indices, val_indices) tuples — length n_splits
    all_patients : sorted list of patient directory paths
    """
    all_patients = []
    for subset in ("train_set", "validation_set"):
        subset_path = os.path.join(root_dir, subset)
        if os.path.isdir(subset_path):
            all_patients.extend(
                sorted(glob.glob(os.path.join(subset_path, "patient*")))
            )

    if not all_patients:
        raise RuntimeError(
            f"No patient directories found under '{root_dir}'.\n"
            "Expected subdirectories 'train_set' and/or 'validation_set' "
            "each containing 'patientXXX' folders."
        )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(all_patients)), all_patients


# ---------------------------------------------------------------------------
# Dataset 1: slice-level dataset for UNet segmentation training
# ---------------------------------------------------------------------------

class ACDC_Dataset(Dataset):
    """
    Each item is one 2-D slice from an ED or ES 3-D volume.
    Used by train_seg.py (UNet training).
    """

    def __init__(self, root_dir, fold=0, mode="train", transform=None):
        splits, all_patients = get_patient_splits(root_dir)
        train_idx, val_idx   = splits[fold]
        target_indices       = train_idx if mode == "train" else val_idx
        self.patient_dirs    = [all_patients[i] for i in target_indices]
        self.transform       = transform
        self.data_items      = []

        for p_dir in self.patient_dirs:
            p_id      = os.path.basename(p_dir)
            info_file = os.path.join(p_dir, "Info.cfg")
            diagnosis = self._parse_diagnosis(info_file)

            all_files = os.listdir(p_dir)
            images = sorted([
                f for f in all_files
                if f.endswith(".nii.gz") and "_gt" not in f and "_4d" not in f
            ])
            masks = sorted([f for f in all_files if f.endswith("_gt.nii.gz")])

            if len(images) == 2 and len(masks) == 2:
                for i in range(2):
                    img_path   = os.path.join(p_dir, images[i])
                    img_itk    = sitk.ReadImage(img_path)
                    n_slices   = img_itk.GetSize()[2]
                    frame_type = "ED" if "frame01" in images[i] else "ES"
                    for s in range(n_slices):
                        self.data_items.append({
                            "patient_id":  p_id,
                            "image_path":  img_path,
                            "mask_path":   os.path.join(p_dir, masks[i]),
                            "diagnosis":   diagnosis,
                            "frame_type":  frame_type,
                            "slice_idx":   s,
                        })

    # ------------------------------------------------------------------
    def _parse_diagnosis(self, info_file):
        if not os.path.isfile(info_file):
            return "UNKNOWN"
        with open(info_file) as f:
            for line in f:
                if line.startswith("Group:"):
                    return line.split(":")[1].strip()
        return "UNKNOWN"

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item    = self.data_items[idx]
        img_itk = sitk.ReadImage(item["image_path"])
        msk_itk = sitk.ReadImage(item["mask_path"])

        image = sitk.GetArrayFromImage(img_itk)[item["slice_idx"]].astype(np.float32)
        mask  = sitk.GetArrayFromImage(msk_itk)[item["slice_idx"]].astype(np.float32)

        image = self._pad_or_crop(image, (256, 256))
        mask  = self._pad_or_crop(mask,  (256, 256))

        # Per-slice Z-score normalisation
        image = (image - image.mean()) / (image.std() + 1e-8)

        sample = {
            "image":      image,
            "mask":       mask,
            "diagnosis":  item["diagnosis"],
            "patient_id": item["patient_id"],
            "frame_type": item["frame_type"],
            "spacing":    img_itk.GetSpacing(),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def _pad_or_crop(arr, target):
        th, tw = target
        h, w   = arr.shape
        # Pad if needed
        ph = max(0, th - h); pw = max(0, tw - w)
        arr = np.pad(arr, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
                     mode="constant")
        # Crop if still too large
        h, w = arr.shape
        sh = (h - th) // 2; sw = (w - tw) // 2
        return arr[sh:sh + th, sw:sw + tw]


# ---------------------------------------------------------------------------
# Dataset 2: patient-level dataset for XGBoost clinical feature extraction
# ---------------------------------------------------------------------------

class ACDC_PatientDataset:
    """
    Returns full 3-D ED + ES volumes for one patient.
    Used by train_classifier.py and predict.py.
    """

    def __init__(self, root_dir, fold=0, mode="train"):
        splits, all_patients = get_patient_splits(root_dir)
        train_idx, val_idx   = splits[fold]
        target_indices       = train_idx if mode == "train" else val_idx
        self.patient_dirs    = [all_patients[i] for i in target_indices]

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        p_dir     = self.patient_dirs[idx]
        p_id      = os.path.basename(p_dir)
        info_file = os.path.join(p_dir, "Info.cfg")

        height = weight = 0.0
        diagnosis = "UNKNOWN"
        with open(info_file) as f:
            for line in f:
                if line.startswith("Group:"):
                    diagnosis = line.split(":")[1].strip()
                elif line.startswith("Height:"):
                    height = float(line.split(":")[1].strip())
                elif line.startswith("Weight:"):
                    weight = float(line.split(":")[1].strip())

        # Mosteller BSA formula; fall back to population mean if data missing
        bsa = (np.sqrt((height * weight) / 3600.0)
               if height > 0 and weight > 0 else 1.7)

        all_files = os.listdir(p_dir)
        images = sorted([
            f for f in all_files
            if f.endswith(".nii.gz") and "_gt" not in f and "_4d" not in f
        ])
        masks = sorted([f for f in all_files if f.endswith("_gt.nii.gz")])

        ed_img_itk  = sitk.ReadImage(os.path.join(p_dir, images[0]))
        es_img_itk  = sitk.ReadImage(os.path.join(p_dir, images[1]))
        ed_mask_itk = sitk.ReadImage(os.path.join(p_dir, masks[0]))
        es_mask_itk = sitk.ReadImage(os.path.join(p_dir, masks[1]))

        return {
            "patient_id": p_id,
            "diagnosis":  diagnosis,
            "ed_image":   sitk.GetArrayFromImage(ed_img_itk),
            "es_image":   sitk.GetArrayFromImage(es_img_itk),
            "ed_mask":    sitk.GetArrayFromImage(ed_mask_itk),
            "es_mask":    sitk.GetArrayFromImage(es_mask_itk),
            "spacing":    ed_img_itk.GetSpacing(),
            "bsa":        bsa,
        }


# ---------------------------------------------------------------------------
# Dataset 3: patient-level 2-channel slice dataset for DenseNet training
# ---------------------------------------------------------------------------

class ACDC_DiagnosisDataset(Dataset):
    """
    Returns a 2-channel (ED, ES) mid-ventricular slice pair per patient.
    Used by train_densenet.py.
    """

    def __init__(self, root_dir, fold=0, mode="train"):
        splits, all_patients = get_patient_splits(root_dir)
        train_idx, val_idx   = splits[fold]
        target_indices       = train_idx if mode == "train" else val_idx
        self.patient_dirs    = [all_patients[i] for i in target_indices]
        self.class_map       = config.DIAGNOSIS_MAP

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        p_dir     = self.patient_dirs[idx]
        info_file = os.path.join(p_dir, "Info.cfg")

        diagnosis = "UNKNOWN"
        with open(info_file) as f:
            for line in f:
                if line.startswith("Group:"):
                    diagnosis = line.split(":")[1].strip()

        all_files = os.listdir(p_dir)
        images = sorted([
            f for f in all_files
            if f.endswith(".nii.gz") and "_gt" not in f and "_4d" not in f
        ])

        ed_vol = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(p_dir, images[0]))
        )
        es_vol = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(p_dir, images[1]))
        )

        mid_idx  = ed_vol.shape[0] // 2
        ed_slice = self._preprocess(ed_vol[mid_idx].astype(np.float32))
        es_slice = self._preprocess(es_vol[mid_idx].astype(np.float32))

        # 2-channel input: channel 0 = ED, channel 1 = ES
        combined = np.stack([ed_slice, es_slice], axis=0)   # (2, 256, 256)

        return {
            "image":      torch.from_numpy(combined),
            "label":      self.class_map.get(diagnosis, 0),
            "patient_id": os.path.basename(p_dir),
        }

    @staticmethod
    def _preprocess(sl):
        h, w = sl.shape
        ph   = max(0, 256 - h); pw = max(0, 256 - w)
        sl   = np.pad(sl, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
                      mode="constant")
        sh   = (sl.shape[0] - 256) // 2
        sw   = (sl.shape[1] - 256) // 2
        sl   = sl[sh:sh + 256, sw:sw + 256]
        return (sl - sl.mean()) / (sl.std() + 1e-8)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dataset_path = "/home/tahir/Cardiac_XAI_Pipeline/FINAL/data"
    ds = ACDC_Dataset(root_dir=dataset_path, fold=0, mode="train")
    print(f"Slice-level training items: {len(ds)}")
    if ds:
        s = ds[0]
        print(f"  image shape : {s['image'].shape}")
        print(f"  diagnosis   : {s['diagnosis']}")
