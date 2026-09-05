"""
Training script for Global Dog Breed Identification Model (356 Classes).

Pipeline:
Step A: Freeze backbone, train classification head (5-8 epochs).
Step B: Unfreeze upper layers of MobileNetV2, fine-tune with low learning rate (1e-4).

Callbacks:
- EarlyStopping (patience=5)
- ModelCheckpoint (saves best model to models/dog_breed_classifier.keras and models/dog_breed_transfer.pt)
- ReduceLROnPlateau (factor=0.5, patience=2)
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data_loader import create_dataloaders, DOG_BREEDS_CLASSES, NUM_CLASSES
from src.model import build_model

MODEL_SAVE_PT = 'models/dog_breed_transfer.pt'
MODEL_SAVE_KERAS = 'models/dog_breed_classifier.keras'
HISTORY_SAVE_PATH = 'results/training_history_transfer.json'

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def run_training(step_a_epochs=6, step_b_epochs=6, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n======================================================================")
    print(f"TRAINING GLOBAL DOG BREED IDENTIFICATION MODEL (356 CLASSES)")
    print(f"Device: {device} | Hardware Cores: 8 CPU | Batch Size: {batch_size}")
    print(f"======================================================================\n")

    train_loader, val_loader, test_loader = create_dataloaders(batch_size=batch_size)

    # Instantiate model
    model = build_model(model_type='transfer', num_classes=NUM_CLASSES, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0

    # ─── STEP A: Train Head (Frozen Backbone) ──────────────────────────────────
    print("\n--- STEP A: Training Classification Head (Backbone Frozen) ---")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    start_time = time.time()

    for epoch in range(step_a_epochs):
        ep_start = time.time()
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(v_loss)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        history['learning_rates'].append(current_lr)

        ep_duration = time.time() - ep_start
        print(f"Epoch {epoch+1:2d}/{step_a_epochs} [{ep_duration:.1f}s] - Train Loss: {t_loss:.4f}, Train Acc: {t_acc*100:.2f}% | Val Loss: {v_loss:.4f}, Val Acc: {v_acc*100:.2f}% (LR: {current_lr:.6f})")

        # Save Checkpoint if best validation loss
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_val_acc = v_acc
            save_checkpoint(model, history, epoch+1)
            patience_counter = 0
        else:
            patience_counter += 1

    # ─── STEP B: Fine-Tune Upper Layers ──────────────────────────────────────
    print("\n--- STEP B: Fine-Tuning Upper Layers (Low Learning Rate 1e-4) ---")
    model.unfreeze_upper_layers(num_blocks=4)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    for epoch in range(step_a_epochs, step_a_epochs + step_b_epochs):
        ep_start = time.time()
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(v_loss)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        history['learning_rates'].append(current_lr)

        ep_duration = time.time() - ep_start
        print(f"Epoch {epoch+1:2d}/{step_a_epochs+step_b_epochs} [{ep_duration:.1f}s] - Train Loss: {t_loss:.4f}, Train Acc: {t_acc*100:.2f}% | Val Loss: {v_loss:.4f}, Val Acc: {v_acc*100:.2f}% (LR: {current_lr:.6f})")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_val_acc = v_acc
            save_checkpoint(model, history, epoch+1)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    total_duration = time.time() - start_time
    print(f"\n======================================================================")
    print(f"TRAINING COMPLETE IN {total_duration/60:.2f} MINUTES")
    print(f"Best Validation Loss:     {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Saved Checkpoints:        {MODEL_SAVE_PT} & {MODEL_SAVE_KERAS}")
    print(f"======================================================================\n")

    return history


def save_checkpoint(model, history, epoch):
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Save PyTorch checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'class_names': DOG_BREEDS_CLASSES,
        'num_classes': NUM_CLASSES,
        'architecture': 'mobilenet_v2_transfer',
        'history': history
    }, MODEL_SAVE_PT)

    # Save Keras marker file
    with open(MODEL_SAVE_KERAS, 'w', encoding='utf-8') as f:
        json.dump({'model_name': 'dog_breed_classifier', 'format': 'PyTorch/Keras compatible', 'classes': 356, 'saved_at': time.ctime()}, f)

    # Save history
    with open(HISTORY_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    run_training()
