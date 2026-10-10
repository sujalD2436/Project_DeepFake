import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.kyc_demo.config import MODEL_PATH
from src.kyc_demo.model import image_transform, load_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, default=Path("data"))
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=MODEL_PATH,
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = datasets.ImageFolder(args.data / "val", transform=image_transform())

if dataset.class_to_idx != {"fake": 0, "real": 1}:
    raise ValueError(f"Unexpected class folders: {dataset.class_to_idx}")

loader = DataLoader(dataset, batch_size=16, shuffle=False)
model, _ = load_checkpoint(args.checkpoint, device)

# Rows are actual labels; columns are predicted labels.
# Order: fake, real
matrix = [[0, 0], [0, 0]]

with torch.inference_mode():
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        for actual, predicted in zip(labels, predictions):
            matrix[int(actual)][int(predicted)] += 1

fake_total = sum(matrix[0])
real_total = sum(matrix[1])
total = fake_total + real_total
correct = matrix[0][0] + matrix[1][1]

print(f"Device: {device}")
print(f"Overall accuracy: {correct / total:.1%}")
print(f"Fake recall: {matrix[0][0] / fake_total:.1%}  (fake crops correctly caught)")
print(f"Real recall: {matrix[1][1] / real_total:.1%}  (real crops correctly recognized)")
print("Confusion matrix (actual rows: fake, real; predicted columns: fake, real):")
print(matrix)