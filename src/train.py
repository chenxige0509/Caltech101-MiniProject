"""
Train CNN (ResNet-18, EfficientNet-B0, ViT-B/16) with frozen backbone on Caltech-101.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.config import (
    PROJECT_ROOT,
    SEED,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    set_seed,
    get_device,
)
from src.models import get_model


def _train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CaltechCSVDataset(Dataset):
    def __init__(self, csv_path: Path, root: Path, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.root / row["filepath"]
        img = Image.open(path).convert("RGB")
        y = int(row["label_id"])
        if self.transform:
            img = self.transform(img)
        return img, y


def train(
    model_name: str,
    image_size: int = 128,
    augment: bool = True,
    optimizer_name: str = "adam",
    num_epochs: int = 25,
    num_workers: int = 12,
    batch_size: int = 32,
) -> None:
    set_seed(SEED)
    device = get_device()
    root = PROJECT_ROOT
    train_csv = root / "train.csv"
    val_csv = root / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Need {train_csv} and {val_csv}. Run caltech101_data_split.ipynb first.")

    train_tf = _train_transform(image_size) if augment else _eval_transform(image_size)
    val_tf = _eval_transform(image_size)
    train_ds = CaltechCSVDataset(train_csv, root, transform=train_tf)
    val_ds = CaltechCSVDataset(val_csv, root, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = get_model(model_name, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "adam":
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    elif optimizer_name == "adamw":
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    elif optimizer_name == "sgd":
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    tag = f"{model_name}_img{image_size}_aug{1 if augment else 0}_{optimizer_name}"
    ckpt_dir = root / "outputs" / "checkpoints"
    log_dir = root / "outputs" / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / f"{tag}_best.pt"
    history_path = log_dir / f"{tag}_history.json"

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=loss.item(), acc=correct / total)
        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, best_ckpt_path)

    history_path.write_text(json.dumps(history, indent=2))
    print(f"  Best val acc: {best_val_acc:.4f}  |  Saved {best_ckpt_path.name}")
