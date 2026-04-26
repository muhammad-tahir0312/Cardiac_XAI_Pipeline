import argparse
import json
import os
import random
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataloader import ACDC_Dataset
from networks import UNet
from visualize import plot_learning_curves

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target):
        inputs         = torch.softmax(inputs, dim=1)
        target_one_hot = (
            torch.eye(self.n_classes, device=inputs.device)[target]
            .permute(0, 3, 1, 2)
        )
        dims         = (0, 2, 3)
        intersection = torch.sum(inputs * target_one_hot, dims)
        cardinality  = torch.sum(inputs + target_one_hot, dims)
        dice_score   = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        return 1.0 - torch.mean(dice_score)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, target):
        ce_loss = F.cross_entropy(
            inputs, target, reduction='none', weight=self.alpha
        )
        pt      = torch.exp(-ce_loss)
        focal   = (1 - pt) ** self.gamma * ce_loss
        return torch.mean(focal)


# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------

def calculate_clinical_metrics(pred, target):
    """
    Per-slice Dice and Hausdorff Distance for labels 1–3 (RV, Myo, LV).
    Returns (mean_dice, mean_hd).
    """
    dice_scores, hd_scores = [], []

    for c in range(1, 4):
        p = (pred   == c).astype(np.uint8)
        t = (target == c).astype(np.uint8)

        if np.sum(t) == 0:
            continue

        dice = 2.0 * np.sum(p * t) / (np.sum(p) + np.sum(t) + 1e-8)
        dice_scores.append(dice)

        if np.sum(p) > 0:
            p_itk = sitk.GetImageFromArray(p)
            t_itk = sitk.GetImageFromArray(t)
            hd_filter = sitk.HausdorffDistanceImageFilter()
            hd_filter.Execute(p_itk, t_itk)
            hd_scores.append(hd_filter.GetHausdorffDistance())
        else:
            hd_scores.append(100.0)

    if not dice_scores:
        return 1.0, 100.0
    return float(np.mean(dice_scores)), float(np.mean(hd_scores))


# ---------------------------------------------------------------------------
# On-the-fly augmentation
# ---------------------------------------------------------------------------

def augment_batch(images, masks):
    """
    Random horizontal flip, vertical flip, and 90°/180°/270° rotation —
    applied consistently to image and mask.
    """
    aug_images, aug_masks = [], []

    for i in range(images.shape[0]):
        img = images[i]   # (1, H, W)
        msk = masks[i]    # (H, W)

        if random.random() > 0.5:
            img = torch.flip(img, dims=[2])   # horizontal
            msk = torch.flip(msk, dims=[1])

        if random.random() > 0.5:
            img = torch.flip(img, dims=[1])   # vertical
            msk = torch.flip(msk, dims=[0])

        k = random.randint(0, 3)
        if k > 0:
            img = torch.rot90(img, k, dims=[1, 2])
            msk = torch.rot90(msk, k, dims=[0, 1])

        aug_images.append(img)
        aug_masks.append(msk)

    return torch.stack(aug_images), torch.stack(aug_masks)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    batch_size   = config.SEG_BATCH_SIZE
    lr           = args.lr     if args.lr     else config.SEG_LR
    epochs       = args.epochs if args.epochs else config.SEG_EPOCHS
    device       = config.DEVICE
    dataset_path = config.DATA_DIR

    os.makedirs(config.MODELS_DIR,  exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    for fold in range(5):
        print(f"\n{'='*45}")
        print(f"   TRAINING UNET  —  FOLD {fold}")
        print(f"{'='*45}")

        # ------------------------------------------------------------------ #
        # Dataset / DataLoader                                                #
        # ------------------------------------------------------------------ #
        train_dataset = ACDC_Dataset(
            root_dir=dataset_path, fold=fold, mode="train"
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=2, pin_memory=True,
        )

        val_dataset = ACDC_Dataset(
            root_dir=dataset_path, fold=fold, mode="val"
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=2, pin_memory=True,
        )

        # ------------------------------------------------------------------ #
        # Model, losses, optimiser                                           #
        # ------------------------------------------------------------------ #
        model = UNet(in_chns=1, class_num=4).to(device)

        class_weights = torch.tensor(
            [0.2, 1.0, 1.0, 1.0], dtype=torch.float32
        ).to(device)
        focal_loss = FocalLoss(gamma=2.0, alpha=class_weights)
        dice_loss  = DiceLoss(n_classes=4)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        fold_model_path    = config.SEG_MODEL_PATH.replace(".pth", f"_fold{fold}.pth")
        history_path       = os.path.join(config.RESULTS_DIR, f"history_fold{fold}.json")
        best_val_dice      = 0.0
        patience_counter   = 0
        early_stop_patience = 15

        history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_hd': []}

        for epoch in range(epochs):
            # --- Training ---
            model.train()
            train_loss = 0.0
            for batch in tqdm(
                train_loader,
                desc=f"Fold {fold} | Epoch {epoch+1}/{epochs} [Train]",
            ):
                images = batch['image'].to(device).unsqueeze(1)
                masks  = batch['mask'].to(device).long()
                images, masks = augment_batch(images, masks)

                optimizer.zero_grad()
                outputs = model(images)
                loss = (0.3 * focal_loss(outputs, masks)
                        + 0.7 * dice_loss(outputs, masks))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            # --- Validation ---
            model.eval()
            val_loss = total_dice = total_hd = n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    images  = batch['image'].to(device).unsqueeze(1)
                    masks   = batch['mask'].to(device).long()
                    outputs = model(images)
                    loss = (0.4 * focal_loss(outputs, masks)
                            + 0.6 * dice_loss(outputs, masks))
                    val_loss += loss.item()

                    preds   = torch.argmax(outputs, dim=1).cpu().numpy()
                    targets = masks.cpu().numpy()
                    for b in range(preds.shape[0]):
                        d, h    = calculate_clinical_metrics(preds[b], targets[b])
                        total_dice += d
                        total_hd   += h
                        n_val      += 1

            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)
            avg_dice  = total_dice / max(n_val, 1)
            avg_hd    = total_hd   / max(n_val, 1)

            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)
            history['val_dice'].append(avg_dice)
            history['val_hd'].append(avg_hd)

            print(f"  Epoch {epoch+1:3d}: "
                  f"Loss(T/V)={avg_train:.3f}/{avg_val:.3f}  "
                  f"Dice={avg_dice:.3f}  HD={avg_hd:.1f}mm")

            scheduler.step(avg_dice)

            if avg_dice > best_val_dice:
                best_val_dice = avg_dice
                torch.save(model.state_dict(), fold_model_path)
                print(f"  --> Saved best model (Dice={best_val_dice:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1

            with open(history_path, 'w') as f:
                json.dump(history, f)

            if patience_counter >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch+1}.")
                break

        # Generate learning-curve plot for this fold
        plot_learning_curves(
            history_path, save_name=f"learning_curves_fold{fold}.png"
        )
        print(f"\n  Fold {fold} best Dice: {best_val_dice:.4f}")

    print("\nAll folds training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train 5-fold UNet segmentation model."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate"
    )
    args = parser.parse_args()
    train(args)
