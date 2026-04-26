import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

import config
from dataloader import ACDC_DiagnosisDataset
from networks import DenseNetDiagnosis

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_densenet(args):
    batch_size   = config.DENSE_BATCH_SIZE
    lr           = args.lr     if args.lr     else config.DENSE_LR
    epochs       = args.epochs if args.epochs else config.DENSE_EPOCHS
    device       = config.DEVICE
    dataset_path = config.DATA_DIR

    os.makedirs(config.MODELS_DIR,  exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    all_fold_acc = []

    for fold in range(5):
        print(f"\n{'='*45}")
        print(f"   TRAINING DENSENET  —  FOLD {fold}")
        print(f"{'='*45}")

        # ------------------------------------------------------------------ #
        # Dataset / DataLoader                                                #
        # ------------------------------------------------------------------ #
        train_dataset = ACDC_DiagnosisDataset(
            root_dir=dataset_path, fold=fold, mode="train"
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=2, pin_memory=True,
        )

        val_dataset = ACDC_DiagnosisDataset(
            root_dir=dataset_path, fold=fold, mode="val"
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=2, pin_memory=True,
        )

        # ------------------------------------------------------------------ #
        # Model, loss, optimiser                                              #
        # ------------------------------------------------------------------ #
        model     = DenseNetDiagnosis(class_num=5, pretrained=True).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        best_val_acc    = 0.0
        fold_model_path = config.DENSE_MODEL_PATH.replace(
            ".pth", f"_fold{fold}.pth"
        )

        for epoch in range(epochs):
            # --- Training ---
            model.train()
            train_loss = 0.0
            for batch in tqdm(
                train_loader,
                desc=f"Fold {fold} | Epoch {epoch+1}/{epochs} [Train]",
            ):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # --- Validation ---
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(images)
                    preds   = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

            val_acc = accuracy_score(all_labels, all_preds)
            print(f"  Epoch {epoch+1:3d}: "
                  f"TrainLoss={avg_train_loss:.4f}  ValAcc={val_acc:.4f}")

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), fold_model_path)
                print(f"  --> Saved best model (ValAcc={best_val_acc:.4f})")

        print(f"\n  Fold {fold} best validation accuracy: {best_val_acc:.4f}")

        all_fold_acc.append(best_val_acc)

    # ---------------------------------------------------------------------- #
    # Final cross-fold summary          #
    # ---------------------------------------------------------------------- #
    print(f"\n{'='*45}")
    print(f"   5-FOLD DENSENET ACCURACY: "
          f"{np.mean(all_fold_acc):.4f} (+/- {np.std(all_fold_acc):.4f})")
    print(f"   Per-fold: {[f'{a:.4f}' for a in all_fold_acc]}")
    print(f"{'='*45}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DenseNet-121 diagnosis model (5-fold CV)."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate"
    )
    args = parser.parse_args()
    train_densenet(args)
