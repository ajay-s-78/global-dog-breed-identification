"""
Inference & Prediction Module for Global Dog Breed Identification.
Loads trained 356-class model and classifies uploaded images into 356 dog breed classes.
"""

import os
import json
import torch
import torch.nn.functional as F
from PIL import Image

from src.data_loader import (
    get_inference_transforms, DOG_BREEDS_CLASSES, NUM_CLASSES, TRANSFER_IMG_SIZE
)
from src.model import get_model


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_production_model(model_path=None, device=None):
    """
    Load best trained 356-class dog breed model.
    Always uses the authoritative 356 real dog breed names mapping.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_path is None:
        model_path = _find_best_model()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'. Train first: python -m src.train")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_type = checkpoint.get('model_type', 'transfer')
    num_classes = checkpoint.get('num_classes', NUM_CLASSES)
    img_size = checkpoint.get('img_size', TRANSFER_IMG_SIZE)

    # Use authoritative 356 breed names list (override any generic placeholder in checkpoint)
    ckpt_classes = checkpoint.get('class_names', [])
    if ckpt_classes and len(ckpt_classes) == NUM_CLASSES and not any(name.startswith('Dog Breed ') for name in ckpt_classes[:5]):
        class_names = ckpt_classes
    else:
        class_names = DOG_BREEDS_CLASSES

    model = get_model(model_type, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded 356-Class Dog Breed Model: {model_path}")
    print(f"  Architecture: {model_type}, Classes: {num_classes}")

    return model, model_type, img_size, class_names


def _find_best_model():
    """Find best trained model from models/ directory."""
    comp_path = 'results/model_comparison.json'
    if os.path.exists(comp_path):
        with open(comp_path, 'r', encoding='utf-8') as f:
            comp = json.load(f)
        best = comp.get('best_model', '').lower()
        if 'scratch' in best and os.path.exists('models/dog_breed_scratch.pt'):
            return 'models/dog_breed_scratch.pt'

    if os.path.exists('models/dog_breed_transfer.pt'):
        return 'models/dog_breed_transfer.pt'
    if os.path.exists('models/dog_breed_scratch.pt'):
        return 'models/dog_breed_scratch.pt'

    if os.path.exists('models/cnn_transfer.pt'):
        return 'models/cnn_transfer.pt'

    raise FileNotFoundError("No trained dog breed models found in 'models/'. Run: python -m src.train")


# ─── Prediction Function ─────────────────────────────────────────────────────

def predict_image(image, model, img_size=224, device=None, top_k=3, class_names=None):
    """
    Predict the dog breed class of a single image.
    Uses strict zero-based indexing (0 to 355) mapped directly to real breed names.

    Returns:
        dict containing:
          - predicted_class: top real breed name (e.g. 'German Shepherd Dog', 'Golden Retriever')
          - confidence: top confidence float (0 to 1)
          - confidence_pct: formatted percentage string
          - top_predictions: list of (real_breed_name, probability) for Top K
          - non_dog_warning: boolean warning if confidence is unusually low
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if class_names is None or any(name.startswith('Dog Breed ') for name in class_names[:5]):
        class_names = DOG_BREEDS_CLASSES

    if image.mode != 'RGB':
        image = image.convert('RGB')

    transform = get_inference_transforms(img_size)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)

    top_probs, top_indices = torch.topk(probabilities, min(top_k, len(class_names)))
    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()

    if top_probs.ndim == 0:
        top_probs = [float(top_probs)]
        top_indices = [int(top_indices)]

    top_predictions = []
    for idx, prob in zip(top_indices, top_probs):
        idx_int = int(idx)
        if 0 <= idx_int < len(class_names):
            breed_name = class_names[idx_int]
        else:
            raise ValueError(f"Predicted class index {idx_int} is out of bounds for {len(class_names)} classes.")
        top_predictions.append((breed_name, float(prob)))

    predicted_class = top_predictions[0][0]
    confidence = top_predictions[0][1]

    # Heuristic warning: if top confidence is under 15% across 356 classes, image might not contain a recognized dog breed
    non_dog_warning = confidence < 0.15

    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'confidence_pct': f"{confidence * 100:.1f}%",
        'top_predictions': top_predictions,
        'non_dog_warning': non_dog_warning,
    }


def predict_from_path(image_path, model=None, model_path=None):
    """Convenience function for CLI prediction."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: '{image_path}'")

    image = Image.open(image_path)
    if model is None:
        model, model_type, img_size, class_names = load_production_model(model_path)
        device = next(model.parameters()).device
    else:
        device = next(model.parameters()).device
        img_size = 224
        class_names = DOG_BREEDS_CLASSES

    return predict_image(image, model, img_size, device, top_k=3, class_names=class_names)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict Dog Breed for 356 Classes')
    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    result = predict_from_path(args.image, model_path=args.model)
    print("\n" + "="*60)
    print("Global Dog Breed Identification — Prediction")
    print("="*60)
    print(f"Predicted Breed: {result['predicted_class']}")
    print(f"Confidence:      {result['confidence_pct']}")
    print("\nTop 3 Predictions:")
    for i, (breed, prob) in enumerate(result['top_predictions'], 1):
        print(f"  {i}. {breed:<35} {prob*100:5.1f}%")
