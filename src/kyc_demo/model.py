from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from .config import IMAGE_SIZE


def make_model(num_classes: int = 2):
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def image_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_to_idx = checkpoint.get("class_to_idx", {})
    if set(class_to_idx) != {"fake", "real"}:
        raise ValueError("Checkpoint must contain the 'fake' and 'real' classes.")

    model = make_model(num_classes=len(class_to_idx))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_to_idx


def bgr_crop_to_tensor(crop_bgr, transform=None):
    import cv2

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return (transform or image_transform())(Image.fromarray(rgb))

