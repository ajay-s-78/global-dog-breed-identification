"""
Dog / Non-Dog Detection Module for Global Dog Breed Identification.

Pre-screens uploaded images using PyTorch pretrained ImageNet MobileNetV2 classifier.
Determines whether an image contains a dog before running the 356-breed fine-grained classifier.
ImageNet class indices 151 through 268 correspond to all 118 recognized dog breeds and wild canines.
"""

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# ImageNet Dog Class Range: Index 151 (Chihuahua) to Index 268 (Timber wolf)
DOG_START_INDEX = 151
DOG_END_INDEX = 268

_detector_model = None
_device = None

def get_dog_detector_model():
    """Lazy load the pretrained ImageNet MobileNetV2 detector once."""
    global _detector_model, _device
    if _detector_model is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            _detector_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).to(_device)
        except Exception:
            _detector_model = models.mobilenet_v2(pretrained=True).to(_device)
        _detector_model.eval()
    return _detector_model, _device


_detector_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def detect_dog(image, threshold=0.08, top_k=5):
    """
    Analyzes an uploaded PIL Image to determine if a dog is present.

    Returns:
        dict:
            - is_dog (bool): True if a clear dog is present, False otherwise.
            - dog_confidence (float): Estimated confidence that a dog is present (0.0 to 1.0).
            - confidence_pct (str): Formatted percentage string.
            - message (str): User-facing status message.
            - top1_idx (int): ImageNet class index.
    """
    model, device = get_dog_detector_model()

    if image.mode != 'RGB':
        image = image.convert('RGB')

    tensor = _detector_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze()

    top_probs, top_indices = torch.topk(probs, top_k)
    top_indices_list = [int(idx.item()) for idx in top_indices]
    top1_prob = float(top_probs[0].item())

    # Check if any top-K ImageNet predictions are in the dog class range (151..268)
    is_topk_dog = any(DOG_START_INDEX <= idx <= DOG_END_INDEX for idx in top_indices_list)

    # Sum probabilities across all 118 ImageNet dog categories
    dog_probs_sum = float(torch.sum(probs[DOG_START_INDEX:DOG_END_INDEX + 1]).item())

    # Image contains a dog if any top-K ImageNet class is a dog OR total dog probability >= threshold
    is_dog = is_topk_dog or (dog_probs_sum >= threshold)
    confidence = dog_probs_sum if is_topk_dog else max(dog_probs_sum, top1_prob if is_topk_dog else 0.0)

    if is_dog:
        message = "Dog detected. Proceeding to 356-breed classification."
    else:
        message = "No dog detected. Please upload a clear image of a dog."

    return {
        'is_dog': is_dog,
        'dog_confidence': confidence,
        'confidence_pct': f"{confidence * 100:.1f}%",
        'dog_probability_sum': round(dog_probs_sum, 4),
        'message': message,
        'top1_idx': top_indices_list[0]
    }
