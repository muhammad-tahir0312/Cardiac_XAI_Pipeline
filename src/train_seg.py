import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import json

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
    # pred, target are (H, W) numpy arrays
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
            hd_scores.append(100.0) # Penalty for no prediction
    if len(dice_scores) == 0:
        return 1.0, 100.0 # Return poor scores if nothing found
            
    return np.mean(dice_scores), np.mean(hd_scores)

def train():
    # Hyperparameters
    batch_size = config.SEG_BATCH_SIZE
    lr = config.SEG_LR
    epochs = config.SEG_EPOCHS
    device = config.DEVICE
    
    # Dataset
    dataset_path = config.DATA_DIR
    train_dataset = ACDC_Dataset(root_dir=dataset_path, split="train_set")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = ACDC_Dataset(root_dir=dataset_path, split="validation_set")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = UNet(in_chns=1, class_num=4).to(device)
    
    # Hybrid Loss (CE + Dice)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=4)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    print(f"Starting Fine-tuned Segmentation Training on {device}...")
    best_val_loss = float('inf')
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
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = 0.5 * ce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
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
                
                loss = 0.5 * ce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)
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
        with open(os.path.join(config.RESULTS_DIR, "seg_history.json"), 'w') as f:
            json.dump(history, f)
            
        print(f"Epoch {epoch+1}: Loss(T/V): {avg_train_loss:.3f}/{avg_val_loss:.3f} | Dice: {avg_dice:.3f} | HD: {avg_hd:.1f}mm")
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = config.SEG_MODEL_PATH
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved Best Model (Val Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    train()
