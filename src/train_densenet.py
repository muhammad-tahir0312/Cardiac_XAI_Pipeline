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
    batch_size = config.DENSE_BATCH_SIZE
    lr = args.lr if args.lr else config.DENSE_LR
    epochs = args.epochs if args.epochs else config.DENSE_EPOCHS
    device = config.DEVICE
    dataset_path = config.DATA_DIR
    
    for fold in range(5):
        print(f"\n" + "="*40)
        print(f"   TRAINING DENSENET FOLD {fold}")
        print(f"="*40)
        
        # Dataset
        train_dataset = ACDC_DiagnosisDataset(root_dir=dataset_path, fold=fold, mode="train")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = ACDC_DiagnosisDataset(root_dir=dataset_path, fold=fold, mode="val")
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Model
        model = DenseNetDiagnosis(class_num=5).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        best_val_acc = 0.0
        fold_model_path = config.DENSE_MODEL_PATH.replace(".pth", f"_fold{fold}.pth")
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{epochs}"):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
            
            val_acc = accuracy_score(all_labels, all_preds)
            print(f"Epoch {epoch+1}: Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), fold_model_path)
                print(f"--> Saved Best Multi-View DenseNet for Fold {fold} (Acc: {best_val_acc:.4f})")
    
    print(f"\nAll Folds DenseNet Training Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    args = parser.parse_args()
    train_densenet(args)
