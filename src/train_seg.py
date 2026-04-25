import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import json
import random
import argparse

from dataloader import ACDC_Dataset
from networks import UNet
import config
import SimpleITK as sitk

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        target_one_hot = torch.eye(self.n_classes, device=inputs.device)[target].permute(0, 3, 1, 2)
        
        dims = (0, 2, 3)
        intersection = torch.sum(inputs * target_one_hot, dims)
        cardinality = torch.sum(inputs + target_one_hot, dims)
        
        dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6)
        return 1. - torch.mean(dice_score)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, target):
        ce_loss = F.cross_entropy(inputs, target, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return torch.mean(focal_loss)

def calculate_clinical_metrics(pred, target):
    """
    Calculate Dice and Hausdorff Distance for each class using SimpleITK
    """
    dice_scores = []
    hd_scores = []
    
    for c in range(1, 4): # Labels 1, 2, 3 (RV, Myo, LV)
        p = (pred == c).astype(np.uint8)
        t = (target == c).astype(np.uint8)
        
        if np.sum(t) == 0:
            continue
            
        # Dice
        dice = 2.0 * np.sum(p * t) / (np.sum(p) + np.sum(t) + 1e-8)
        dice_scores.append(dice)
        
        # Hausdorff Distance (using SimpleITK)
        if np.sum(p) > 0:
            p_itk = sitk.GetImageFromArray(p)
            t_itk = sitk.GetImageFromArray(t)
            hd_filter = sitk.HausdorffDistanceImageFilter()
            hd_filter.Execute(p_itk, t_itk)
            hd_scores.append(hd_filter.GetHausdorffDistance())
        else:
            hd_scores.append(100.0)
    if len(dice_scores) == 0:
        return 1.0, 100.0
            
    return np.mean(dice_scores), np.mean(hd_scores)

def augment_batch(images, masks):
    """
    On-the-fly data augmentation for cardiac MRI slices.
    """
    augmented_images = []
    augmented_masks = []
    
    for i in range(images.shape[0]):
        img = images[i]   # (1, H, W)
        msk = masks[i]    # (H, W)
        
        if random.random() > 0.5:
            img = torch.flip(img, dims=[2])
            msk = torch.flip(msk, dims=[1])
        
        if random.random() > 0.5:
            img = torch.flip(img, dims=[1])
            msk = torch.flip(msk, dims=[0])
        
        k = random.randint(0, 3)
        if k > 0:
            img = torch.rot90(img, k, dims=[1, 2])
            msk = torch.rot90(msk, k, dims=[0, 1])
        
        augmented_images.append(img)
        augmented_masks.append(msk)
    
    return torch.stack(augmented_images), torch.stack(augmented_masks)

def train(args):
    batch_size = config.SEG_BATCH_SIZE
    lr = args.lr if args.lr else config.SEG_LR
    epochs = args.epochs if args.epochs else config.SEG_EPOCHS
    device = config.DEVICE

    # Run 5-Fold Cross-Validation
    for fold in range(5):
        print(f"\n" + "="*40)
        print(f"   TRAINING FOLD {fold}")
        print(f"="*40)
        
        # Dataset
        dataset_path = config.DATA_DIR
        train_dataset = ACDC_Dataset(root_dir=dataset_path, fold=fold, mode="train")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=2, pin_memory=True)
        
        val_dataset = ACDC_Dataset(root_dir=dataset_path, fold=fold, mode="val")
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)
        
        # Model
        model = UNet(in_chns=1, class_num=4).to(device)
        
        # Loss and Optimizer
        class_weights = torch.tensor([0.2, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)
        focal_loss = FocalLoss(gamma=2.0, alpha=class_weights)
        dice_loss = DiceLoss(n_classes=4)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        best_val_dice = 0.0
        fold_model_path = config.SEG_MODEL_PATH.replace(".pth", f"_fold{fold}.pth")
        
        history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_hd': []}
        patience_counter = 0
        early_stop_patience = 15

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{epochs} [Train]"):
                images = batch['image'].to(device).unsqueeze(1)
                masks = batch['mask'].to(device).long()
                images, masks = augment_batch(images, masks)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = 0.3 * focal_loss(outputs, masks) + 0.7 * dice_loss(outputs, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss, total_dice, total_hd, num_val = 0, 0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device).unsqueeze(1)
                    masks = batch['mask'].to(device).long()
                    outputs = model(images)
                    loss = 0.4 * focal_loss(outputs, masks) + 0.6 * dice_loss(outputs, masks)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    targets = masks.cpu().numpy()
                    for b in range(preds.shape[0]):
                        d, h = calculate_clinical_metrics(preds[b], targets[b])
                        total_dice += d
                        total_hd += h
                        num_val += 1
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_dice = total_dice / num_val
            avg_hd = total_hd / num_val
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_dice'].append(avg_dice)
            history['val_hd'].append(avg_hd)
            
            print(f"Epoch {epoch+1}: Loss(T/V): {avg_train_loss:.3f}/{avg_val_loss:.3f} | Dice: {avg_dice:.3f} | HD: {avg_hd:.1f}mm")
            
            scheduler.step(avg_dice)
            
            if avg_dice > best_val_dice:
                best_val_dice = avg_dice
                torch.save(model.state_dict(), fold_model_path)
                print(f"--> Saved Best Model for Fold {fold} (Dice: {best_val_dice:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    print(f"\nAll Folds Training Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    args = parser.parse_args()
    train(args)
