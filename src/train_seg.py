import torch
import torch.nn as nn
import torch.optim as optim
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
    Applies random horizontal/vertical flips and 90-degree rotations.
    """
    augmented_images = []
    augmented_masks = []
    
    for i in range(images.shape[0]):
        img = images[i]   # (1, H, W)
        msk = masks[i]    # (H, W)
        
        # Random horizontal flip
        if random.random() > 0.5:
            img = torch.flip(img, dims=[2])
            msk = torch.flip(msk, dims=[1])
        
        # Random vertical flip
        if random.random() > 0.5:
            img = torch.flip(img, dims=[1])
            msk = torch.flip(msk, dims=[0])
        
        # Random 90-degree rotation (0, 1, 2, or 3 times)
        k = random.randint(0, 3)
        if k > 0:
            img = torch.rot90(img, k, dims=[1, 2])
            msk = torch.rot90(msk, k, dims=[0, 1])
        
        # Random intensity jitter (brightness/contrast for images only)
        if random.random() > 0.5:
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            img = img * contrast + (brightness - 1.0)
            img = torch.clamp(img, 0, 1)
        
        augmented_images.append(img)
        augmented_masks.append(msk)
    
    return torch.stack(augmented_images), torch.stack(augmented_masks)

def train(args):
    # Hyperparameters
    batch_size = config.SEG_BATCH_SIZE
    lr = args.lr if args.lr else config.SEG_LR
    epochs = args.epochs if args.epochs else config.SEG_EPOCHS
    device = config.DEVICE
    
    # Dataset
    dataset_path = config.DATA_DIR
    train_dataset = ACDC_Dataset(root_dir=dataset_path, split="train_set")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=2, pin_memory=True)
    
    val_dataset = ACDC_Dataset(root_dir=dataset_path, split="validation_set")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    
    # Model
    model = UNet(in_chns=1, class_num=4).to(device)
    
    # Hybrid Loss (CE + Dice) with class weighting for imbalanced cardiac structures
    # Background is huge, RV/Myo/LV are small — weight them higher
    class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(n_classes=4)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    print(f"Starting Fine-tuned Segmentation Training on {device}...")
    print(f"  Batch Size: {batch_size} | LR: {lr} | Epochs: {epochs}")
    print(f"  Augmentation: ON | Class Weights: ON | Scheduler: CosineAnnealing")
    
    best_val_dice = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_hd': []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = batch['image'].to(device).unsqueeze(1)
            masks = batch['mask'].to(device).long()
            
            # Apply augmentation
            images, masks = augment_batch(images, masks)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = 0.4 * ce_loss(outputs, masks) + 0.6 * dice_loss(outputs, masks)
            loss.backward()
            
            # Gradient clipping to prevent spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # scheduler.step() is now called after validation with the metric
            
        # Validation
        model.eval()
        val_loss = 0
        total_dice = 0
        total_hd = 0
        num_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device).unsqueeze(1)
                masks = batch['mask'].to(device).long()
                outputs = model(images)
                
                loss = 0.4 * ce_loss(outputs, masks) + 0.6 * dice_loss(outputs, masks)
                val_loss += loss.item()
                
                # Clinical Metrics
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
        
        # Update History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_dice'].append(avg_dice)
        history['val_hd'].append(avg_hd)
        
        # Save History JSON
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        with open(os.path.join(config.RESULTS_DIR, "seg_history.json"), 'w') as f:
            json.dump(history, f)
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Loss(T/V): {avg_train_loss:.3f}/{avg_val_loss:.3f} | Dice: {avg_dice:.3f} | HD: {avg_hd:.1f}mm | LR: {current_lr:.6f}")
        
        # Step the scheduler based on validation Dice
        scheduler.step(avg_dice)
        
        # Save best model (track by Dice, not loss — more clinically meaningful)
        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            best_val_loss = avg_val_loss
            save_path = config.SEG_MODEL_PATH
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved Best Model (Dice: {best_val_dice:.4f}, HD: {avg_hd:.1f}mm)")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}. Best Dice: {best_val_dice:.4f}")
            break
    
    print(f"\nTraining Complete! Best Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    args = parser.parse_args()
    train(args)
