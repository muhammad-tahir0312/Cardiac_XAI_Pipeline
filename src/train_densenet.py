import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from dataloader import ACDC_Dataset
from networks import DenseNetDiagnosis
import config

def train_densenet():
    # Hyperparameters
    batch_size = config.DENSE_BATCH_SIZE
    lr = config.DENSE_LR
    epochs = config.DENSE_EPOCHS
    device = config.DEVICE
    
    # Dataset (slice-based for DenseNet)
    dataset_path = config.DATA_DIR
    train_dataset = ACDC_Dataset(root_dir=dataset_path, split="train_set")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Class mapping for diagnosis
    class_map = {"NOR": 0, "MINF": 1, "DCM": 2, "HCM": 3, "ARV": 4}
    
    # Model, Loss, Optimizer
    model = DenseNetDiagnosis(class_num=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting DenseNet training on {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader):
            images = batch['image'].to(device).unsqueeze(1) # (B, 1, H, W)
            
            # Map diagnosis strings to labels
            labels = torch.tensor([class_map.get(d, 0) for d in batch['diagnosis']]).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
        
        # Save model
        save_path = config.DENSE_MODEL_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train_densenet()
