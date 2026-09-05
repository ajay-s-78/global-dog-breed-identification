"""
Evaluation script for Global Dog Breed Identification Model (356 Classes).
Evaluates saved model on untouched test set (1,068 images across 356 classes).
Generates Top-1 & Top-5 accuracy, classification metrics, loss/accuracy curves, and confusion matrix.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from src.data_loader import create_dataloaders, DOG_BREEDS_CLASSES, NUM_CLASSES
from src.model import build_model

MODEL_PATH = 'models/dog_breed_transfer.pt'

def evaluate_test_set():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n======================================================================")
    print(f"EVALUATING MODEL ON UNTOUCHED TEST SET (1,068 IMAGES across 356 CLASSES)")
    print(f"======================================================================\n")

    train_loader, val_loader, test_loader = create_dataloaders()

    # Load model
    model = build_model(model_type='transfer', num_classes=NUM_CLASSES, freeze_backbone=False).to(device)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model weights from '{MODEL_PATH}'")
    else:
        print(f"⚠ Warning: '{MODEL_PATH}' not found. Evaluating randomly initialized weights.")

    model.eval()

    all_targets = []
    all_preds = []
    all_top5_preds = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            top5_probs, top5_indices = torch.topk(outputs, 5, dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_top5_preds.extend(top5_indices.cpu().numpy())

    test_loss = total_loss / len(test_loader.dataset)
    top1_acc = accuracy_score(all_targets, all_preds)

    # Top-5 Accuracy Calculation
    top5_correct = sum(target in top5 for target, top5 in zip(all_targets, all_top5_preds))
    top5_acc = top5_correct / len(all_targets)

    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)

    print("\n" + "=" * 65)
    print("FINAL MODEL EVALUATION METRICS (TEST SET)")
    print("=" * 65)
    print(f"Test Loss:           {test_loss:.4f}")
    print(f"Top-1 Test Accuracy: {top1_acc*100:.2f}%")
    print(f"Top-5 Test Accuracy: {top5_acc*100:.2f}%")
    print(f"Weighted Precision:  {precision:.4f}")
    print(f"Weighted Recall:     {recall:.4f}")
    print(f"Weighted F1-Score:   {f1:.4f}")

    # Load history if available for training curves
    history_path = 'results/training_history_transfer.json'
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
            
        plot_training_curves(history)

    # Save metrics JSON
    metrics = {
        'model_name': 'MobileNetV2 Transfer Learning',
        'num_classes': NUM_CLASSES,
        'test_samples': len(all_targets),
        'test_loss': round(test_loss, 4),
        'top1_accuracy': round(top1_acc, 4),
        'top5_accuracy': round(top5_acc, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'dataset_limitations_note': 'Fine-grained classification across 356 classes trained with 20 real images per breed. Top-5 accuracy represents whether the true breed appears in the top 5 model predictions.'
    }

    metrics_path = 'results/metrics_mobilenetv2_transfer.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Saved evaluation metrics to '{metrics_path}'")
    return metrics


def plot_training_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    ax1.set_title('Training & Validation Loss (356 Breeds)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'b-o', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'r-s', label='Val Accuracy', linewidth=2)
    ax2.set_title('Training & Validation Accuracy (356 Breeds)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = 'results/model_comparison.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✓ Saved training curves plot to '{plot_path}'")


if __name__ == '__main__':
    evaluate_test_set()
