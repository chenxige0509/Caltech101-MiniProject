"""
Evaluate trained CNN on test set: accuracy, top-5 accuracy, classification report.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from src.config import PROJECT_ROOT, NUM_CLASSES, get_device
from src.train import CaltechCSVDataset, _eval_transform
from src.models import get_model


def evaluate_model(
    model_name: str,
    image_size: int,
    augment: bool,
    optimizer_name: str,
    num_workers: int = 12,
    batch_size: int = 32,
) -> dict:
    """
    Load best checkpoint, evaluate on test set. Return dict with test_accuracy, top5_accuracy, etc.
    """
    device = get_device()
    root = PROJECT_ROOT
    test_csv = root / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError("test.csv not found. Run caltech101_data_split.ipynb first.")

    tag = f"{model_name}_img{image_size}_aug{1 if augment else 0}_{optimizer_name}"
    ckpt_path = root / "outputs" / "checkpoints" / f"{tag}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = get_model(model_name, num_classes=NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    test_ds = CaltechCSVDataset(test_csv, root, transform=_eval_transform(image_size))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    test_acc = (preds == labels).mean().item()
    # Top-5 accuracy
    top5_idx = np.argsort(-probs, axis=1)[:, :5]
    top5_acc = np.mean([labels[i] in top5_idx[i] for i in range(len(labels))])

    log_dir = root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / f"{tag}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(labels, preds, digits=4))
    results_path = log_dir / f"{tag}_results.json"
    results = {
        "model": model_name,
        "image_size": image_size,
        "augment": augment,
        "optimizer_name": optimizer_name,
        "test_accuracy": float(test_acc),
        "top5_accuracy": float(top5_acc),
    }
    results_path.write_text(json.dumps(results, indent=2))

    return results

