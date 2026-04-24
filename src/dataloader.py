import os
import glob
import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset
import torch

# Silence SimpleITK warnings about NIfTI scales
sitk.ProcessObject.SetGlobalWarningDisplay(False)

class ACDC_Dataset(Dataset):
    """
    Standard dataset for 3D frames (one item per frame)
    """
    def __init__(self, root_dir, split="train_set", transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.patient_dirs = sorted(glob.glob(os.path.join(self.root_dir, "patient*")))
        self.transform = transform
        self.data_items = []
        
        for p_dir in self.patient_dirs:
            p_id = os.path.basename(p_dir)
            info_file = os.path.join(p_dir, "Info.cfg")
            diagnosis = self._parse_info(info_file)
            
            all_files = os.listdir(p_dir)
            images = sorted([f for f in all_files if f.endswith(".nii.gz") and "_gt" not in f and "_4d" not in f])
            masks = sorted([f for f in all_files if f.endswith("_gt.nii.gz")])
            
            if len(images) == 2 and len(masks) == 2:
                for i in range(2):
                    img_itk = sitk.ReadImage(os.path.join(p_dir, images[i]))
                    num_slices = img_itk.GetSize()[2]
                    frame_type = "ED" if "frame01" in images[i] else "ES"
                    for s in range(num_slices):
                        self.data_items.append({
                            "patient_id": p_id,
                            "image_path": os.path.join(p_dir, images[i]),
                            "mask_path": os.path.join(p_dir, masks[i]),
                            "diagnosis": diagnosis,
                            "frame_type": frame_type,
                            "slice_idx": s
                        })

    def _parse_info(self, info_file):
        diagnosis = "UNKNOWN"
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                for line in f:
                    if line.startswith("Group:"):
                        diagnosis = line.split(":")[1].strip()
        return diagnosis

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        img_itk = sitk.ReadImage(item["image_path"])
        mask_itk = sitk.ReadImage(item["mask_path"])
        
        # Get specific slice
        image = sitk.GetArrayFromImage(img_itk)[item["slice_idx"]].astype(np.float32)
        mask = sitk.GetArrayFromImage(mask_itk)[item["slice_idx"]].astype(np.float32)
        
        # Resize/Padding to 256x256 to ensure consistent batch size
        # ACDC images vary in size (e.g. 256x216, 216x256, etc.)
        image = self._pad_or_crop(image, (256, 256))
        mask = self._pad_or_crop(mask, (256, 256))
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        
        sample = {
            'image': image, 'mask': mask,
            'diagnosis': item["diagnosis"], 'patient_id': item["patient_id"],
            'frame_type': item["frame_type"], 'spacing': img_itk.GetSpacing()
        }
        if self.transform: sample = self.transform(sample)
        return sample

    def _pad_or_crop(self, array, target_shape):
        h, w = array.shape
        th, tw = target_shape
        
        # Padding
        ph = max(0, th - h)
        pw = max(0, tw - w)
        array = np.pad(array, ((ph//2, ph-ph//2), (pw//2, pw-pw//2)), mode='constant')
        
        # Cropping if larger
        h, w = array.shape
        sh = (h - th) // 2
        sw = (w - tw) // 2
        return array[sh:sh+th, sw:sw+tw]

class ACDC_PatientDataset:
    """
    Dataset that returns both ED and ES frames for a single patient (for diagnosis)
    """
    def __init__(self, root_dir, split="train_set"):
        self.root_dir = os.path.join(root_dir, split)
        self.patient_dirs = sorted(glob.glob(os.path.join(self.root_dir, "patient*")))
        
    def __len__(self):
        return len(self.patient_dirs)
        
    def __getitem__(self, idx):
        p_dir = self.patient_dirs[idx]
        p_id = os.path.basename(p_dir)
        info_file = os.path.join(p_dir, "Info.cfg")
        
        diagnosis = "UNKNOWN"
        with open(info_file, 'r') as f:
            for line in f:
                if line.startswith("Group:"):
                    diagnosis = line.split(":")[1].strip()
        
        all_files = os.listdir(p_dir)
        images = sorted([f for f in all_files if f.endswith(".nii.gz") and "_gt" not in f and "_4d" not in f])
        masks = sorted([f for f in all_files if f.endswith("_gt.nii.gz")])
        
        ed_mask_itk = sitk.ReadImage(os.path.join(p_dir, masks[0]))
        es_mask_itk = sitk.ReadImage(os.path.join(p_dir, masks[1]))
        ed_img_itk = sitk.ReadImage(os.path.join(p_dir, images[0]))
        es_img_itk = sitk.ReadImage(os.path.join(p_dir, images[1]))
        
        return {
            "patient_id": p_id,
            "diagnosis": diagnosis,
            "ed_image": sitk.GetArrayFromImage(ed_img_itk),
            "es_image": sitk.GetArrayFromImage(es_img_itk),
            "ed_mask": sitk.GetArrayFromImage(ed_mask_itk),
            "es_mask": sitk.GetArrayFromImage(es_mask_itk),
            "spacing": ed_img_itk.GetSpacing()
        }

if __name__ == "__main__":
    dataset_path = "/home/tahir/Cardiac_XAI_Pipeline/data"
    dataset = ACDC_Dataset(root_dir=dataset_path, split="train_set")
    print(f"Total training frames found: {len(dataset)}")
    if len(dataset) > 0:
        print(f"Sample Diagnosis: {dataset[0]['diagnosis']}")
