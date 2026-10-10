from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.kyc_demo.config import IMAGE_SIZE, MODEL_PATH


def loader(root: Path, training: bool, batch_size: int) -> DataLoader:
    ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        ops += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.1, 0.1, 0.1)]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    dataset = datasets.ImageFolder(root, transform=transforms.Compose(ops))
    if dataset.class_to_idx != {"fake": 0, "real": 1}:
        raise ValueError(f"Expected fake/ and real/ folders under {root}; found {dataset.class_to_idx}")
    if not dataset:
        raise ValueError(f"No images found under {root}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=0)


def epoch(model, data, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_total = correct = count = 0
    for images, labels in data:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        count += labels.size(0)
        loss_total += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    return loss_total / count, correct / count


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV3 on face-crop folders.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Select CPU/GPU explicitly, or choose automatically.")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    train_data = loader(args.data / "train", True, args.batch_size)
    val_data = loader(args.data / "val", False, args.batch_size)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was selected, but CUDA is not available.")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    weights = None if args.no_pretrained else MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    targets = torch.tensor(train_data.dataset.targets, dtype=torch.long)
    class_counts = torch.bincount(targets, minlength=2).float()
    if torch.any(class_counts == 0):
        raise ValueError("Training data must contain at least one fake and one real crop.")
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    best_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Device: {device}; classes: {train_data.dataset.class_to_idx}; "
        f"class crop counts: fake={int(class_counts[0])}, real={int(class_counts[1])}; "
        f"loss weights: fake={class_weights[0]:.3f}, real={class_weights[1]:.3f}"
    )
    for step in range(1, args.epochs + 1):
        tr_loss, tr_acc = epoch(model, train_data, loss_fn, device, optimizer)
        va_loss, va_acc = epoch(model, val_data, loss_fn, device)
        print(f"epoch {step}/{args.epochs}: train loss={tr_loss:.4f} acc={tr_acc:.1%}; val loss={va_loss:.4f} acc={va_acc:.1%}")
        if va_loss < best_loss:
            best_loss = va_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_data.dataset.class_to_idx,
                "image_size": IMAGE_SIZE,
            }, args.output)
    print(f"Saved best validation-loss checkpoint: {args.output}")


if __name__ == "__main__":
    main()
