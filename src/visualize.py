import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import config

# ---------------------------------------------------------------------------
# Learning curves (written by train_seg.py)
# ---------------------------------------------------------------------------

def plot_learning_curves(history_path, save_name="learning_curves.png"):
    """
    Plot Train/Val Loss, Dice, and Hausdorff Distance from a history JSON.

    Parameters
    ----------
    history_path : str  — path to the JSON file written by train_seg.py
    save_name    : str  — output PNG filename (placed in RESULTS_DIR)
    """
    if not os.path.exists(history_path):
        print(f"[visualize] History file not found: {history_path}")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   label='Val Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(epochs, history['val_dice'], color='green', label='Val Dice')
    axes[1].set_title('Validation Dice Score')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].legend()

    axes[2].plot(epochs, history['val_hd'], color='red', label='Val HD')
    axes[2].set_title('Validation Hausdorff Distance')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('HD (mm)')
    axes[2].legend()

    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, save_name)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Learning curves saved to {save_path}")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_diagnosis_metrics(y_true, y_pred, labels, save_name="confusion_matrix.png"):
    """
    Normalised confusion matrix for the diagnosis task.

    Parameters
    ----------
    y_true    : list/array of ground-truth string labels
    y_pred    : list/array of predicted string labels
    labels    : ordered list of class name strings
    save_name : output PNG filename
    """
    cm   = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
    ax.set_title("Diagnosis Confusion Matrix")

    save_path = os.path.join(config.RESULTS_DIR, save_name)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Confusion matrix saved to {save_path}")


# ---------------------------------------------------------------------------
# t-SNE feature space
# ---------------------------------------------------------------------------

def plot_feature_tsne(X, y, save_name="feature_tsne.png"):
    """
    2-D t-SNE projection of the clinical feature space, coloured by diagnosis.

    Parameters
    ----------
    X         : numpy array (N, n_features)
    y         : list/array of string labels of length N
    save_name : output PNG filename
    """
    perplexity = min(30, len(X) - 1)
    tsne       = TSNE(n_components=2, random_state=config.RANDOM_SEED,
                      perplexity=perplexity)
    X_emb      = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_emb[:, 0], y=X_emb[:, 1],
        hue=y, palette='tab10', s=100, alpha=0.8,
    )
    plt.title("t-SNE — Clinical Feature Space")
    plt.legend(title="Diagnosis", bbox_to_anchor=(1.05, 1), loc='upper left')

    save_path = os.path.join(config.RESULTS_DIR, save_name)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] t-SNE plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Segmentation side-by-side
# ---------------------------------------------------------------------------

def visualize_segmentation(image, target, pred, save_name="seg_viz.png"):
    """
    Side-by-side: MRI slice | Ground Truth mask | Predicted mask.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image,  cmap='gray'); axes[0].set_title("MRI Slice");         axes[0].axis('off')
    axes[1].imshow(target, cmap='jet');  axes[1].set_title("Ground Truth Mask");  axes[1].axis('off')
    axes[2].imshow(pred,   cmap='jet');  axes[2].set_title("AI Prediction");      axes[2].axis('off')

    tag = save_name.replace("seg_viz_", "").replace(".png", "")
    plt.suptitle(f"Segmentation — {tag}")

    save_path = os.path.join(config.RESULTS_DIR, save_name)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# UNet encoder feature maps
# ---------------------------------------------------------------------------

def visualize_feature_maps(model, input_slice, save_name="feature_maps.png"):
    """
    Display the first 16 activation channels from the UNet encoder's input conv.

    Parameters
    ----------
    model       : trained UNet instance
    input_slice : 2-D numpy array (H, W) — one MRI slice
    save_name   : output PNG filename
    """
    model.eval()
    x = (torch.from_numpy(input_slice.astype(np.float32))
         .unsqueeze(0).unsqueeze(0))

    with torch.no_grad():
        if hasattr(model, 'encoder'):
            features = model.encoder.in_conv(x)
        else:
            features = model.down1(x)

    features = features[0].cpu().numpy()

    plt.figure(figsize=(15, 10))
    for i in range(min(16, features.shape[0])):
        plt.subplot(4, 4, i + 1)
        plt.imshow(features[i], cmap='viridis')
        plt.axis('off')

    plt.suptitle("First-Layer Feature Maps (UNet Encoder)")
    save_path = os.path.join(config.RESULTS_DIR, save_name)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Feature maps saved to {save_path}")


# ---------------------------------------------------------------------------
# Clinical feature boxplots
# ---------------------------------------------------------------------------

def plot_clinical_boxplots(X, y, feature_names, save_name="clinical_distributions.png"):
    """
    Grid of boxplots showing each clinical feature distribution per class.

    Parameters
    ----------
    X             : numpy array (N, n_features)
    y             : list/array of string labels of length N
    feature_names : list of feature name strings
    save_name     : output PNG filename
    """
    df = pd.DataFrame(X, columns=feature_names)
    df['Diagnosis'] = y

    n_feats = len(feature_names)
    cols    = 3
    rows    = (n_feats + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_names):
        sns.boxplot(
            x='Diagnosis', y=col, data=df,
            hue='Diagnosis', palette='Set2',
            legend=False, ax=axes[i],
        )
        axes[i].set_title(f"Distribution of {col}")

    for j in range(n_feats, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, save_name)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Clinical distributions saved to {save_path}")


# ---------------------------------------------------------------------------
# Module self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("visualize.py — all plotting utilities ready.")
