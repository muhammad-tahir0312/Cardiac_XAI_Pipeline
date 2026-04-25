import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import argparse

from dataloader import ACDC_DiagnosisDataset
from networks import DenseNetDiagnosis
import config

def train_densenet(args):
    # Hyperparameters
    batch_size = config.DENSE_BATCH_SIZE
    lr = args.lr if args.lr else config.DENSE_LR
    epochs = args.epochs if args.epochs else config.DENSE_EPOCHS
    device = config.DEVICE
    
    # Dataset (patient-based pair of ED/ES for DenseNet)
    dataset_path = config.DATA_DIR
    train_dataset = ACDC_DiagnosisDataset(root_dir=dataset_path, split="train_set")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = ACDC_DiagnosisDataset(root_dir=dataset_path, split="validation_set")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Class mapping for diagnosis
    class_map = config.DIAGNOSIS_MAP
    
    # Calculate class weights for imbalance
    # We can just iterate once or use a fixed weight if data is balanced enough
    # For ACDC, it's fairly balanced (20 patients per group)
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)
    
    # Model, Loss, Optimizer
    model = DenseNetDiagnosis(class_num=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"Starting Multi-View DenseNet training on {device}...")
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = batch['image'].to(device) # (B, 2, 256, 256)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation Phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}: Loss(T/V): {avg_train_loss:.4f}/{avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = config.DENSE_MODEL_PATH
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved Best Multi-View DenseNet (Acc: {best_val_acc:.4f})")

    print(f"\nMulti-View DenseNet Training Complete! Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    args = parser.parse_args()
    train_densenet(args)
