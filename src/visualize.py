import os
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import config

def plot_learning_curves(history_path, save_name="learning_curves.png"):
    """
    Plot Train/Val Loss, Dice, and HD from history JSON
    """
    if not os.path.exists(history_path):
        print(f"History file {history_path} not found.")
        return
        
    with open(history_path, 'r') as f:
        history = json.load(f)
        
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Dice
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['val_dice'], label='Val Dice', color='green')
    plt.title('Validation Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.legend()
    
    # HD
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_hd'], label='Val HD', color='red')
    plt.title('Validation Hausdorff Distance')
    plt.xlabel('Epochs')
    plt.ylabel('HD (mm)')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.RESULTS_DIR, save_name))
    print(f"Learning curves saved to {os.path.join(config.RESULTS_DIR, save_name)}")

def plot_diagnosis_metrics(y_true, y_pred, labels, save_name="confusion_matrix.png"):
    """
    Plot Confusion Matrix for Disease Diagnosis
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Diagnosis Confusion Matrix")
    plt.savefig(os.path.join(config.RESULTS_DIR, save_name))
    print(f"Confusion matrix saved to {os.path.join(config.RESULTS_DIR, save_name)}")

def plot_feature_tsne(X, y, save_name="feature_tsne.png"):
    """
    Visualize the Clinical Feature Space using t-SNE
    """
    tsne = TSNE(n_components=2, random_state=config.RANDOM_SEED)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y, palette='viridis', s=100)
    plt.title("t-SNE Visualization of Cardiac Clinical Features")
    plt.savefig(os.path.join(config.RESULTS_DIR, save_name))
    print(f"t-SNE plot saved to {os.path.join(config.RESULTS_DIR, save_name)}")

def visualize_segmentation(image, target, pred, save_name="seg_viz.png"):
    """
    Side-by-side visualization of Image, GT, and Prediction
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("MRI Slice")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(target, cmap='jet')
    plt.title("Ground Truth Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap='jet')
    plt.title("AI Prediction")
    plt.axis('off')
    
    # Add a global title with patient ID
    plt.suptitle(f"Segmentation Results for {save_name.replace('seg_viz_', '').replace('.png', '')}")
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.RESULTS_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_feature_maps(model, input_slice, save_name="feature_maps.png"):
    """
    Visualize the internal activation maps of the first UNet layer
    """
    model.eval()
    with torch.no_grad():
        # Get first conv layer output from encoder
        x = input_slice.unsqueeze(0).unsqueeze(0)
        if hasattr(model, 'encoder'):
            features = model.encoder.in_conv(x)
        else:
            # Fallback for other UNet variations
            features = model.down1(x)
            
    features = features[0].cpu().numpy()
    
    plt.figure(figsize=(15, 10))
    for i in range(min(16, features.shape[0])):
        plt.subplot(4, 4, i+1)
        plt.imshow(features[i], cmap='viridis')
        plt.axis('off')
    
    plt.suptitle("Model Internals: First Layer Feature Maps")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.RESULTS_DIR, save_name))
    print(f"Feature maps saved to {os.path.join(config.RESULTS_DIR, save_name)}")

def plot_clinical_boxplots(X, y, feature_names, save_name="clinical_distributions.png"):
    """
    Research-level Data Visualization: Boxplots of clinical features per disease
    """
    df = pd.DataFrame(X, columns=feature_names)
    df['Diagnosis'] = y
    
    num_features = len(feature_names)
    cols = 3
    rows = (num_features + cols - 1) // cols
    
    plt.figure(figsize=(20, 5 * rows))
    for i, col in enumerate(feature_names):
        plt.subplot(rows, cols, i+1)
        sns.boxplot(x='Diagnosis', y=col, data=df, palette='Set2')
        plt.title(f"Distribution of {col}")
        
    plt.tight_layout()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.RESULTS_DIR, save_name))
    print(f"Clinical distributions saved to {os.path.join(config.RESULTS_DIR, save_name)}")

if __name__ == "__main__":
    # Example usage for testing
    print("Visualization module ready.")
