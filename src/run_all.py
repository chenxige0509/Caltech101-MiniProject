"""
Run ALL experiments (CNN ablations + HOG baselines + HOG F1 eval + 24 remaining combos).
Results are saved to outputs/logs/ and outputs/figures/.

Usage:
    python -m src.run_all
    python -m src.run_all --epochs 25          # full training
    python -m src.run_all --epochs 3 --quick   # smoke test (skip ablations)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog as skimage_hog
from sklearn.metrics import accuracy_score, f1_score

from src.config import PROJECT_ROOT, ensure_dirs
from src.models import VIT_REQUIRED_SIZE

# ---------------------------------------------------------------------------
# CNN configuration
# ---------------------------------------------------------------------------
CNN_MODELS = ["resnet18", "efficientnet_b0", "vit_b16"]
DEFAULT_IMG_SIZE = 128

ABLATIONS = [
    ("baseline", dict()),
    ("img64",    dict(image_size=64)),
    ("img224",   dict(image_size=224)),
    ("no_augment", dict(augment=False)),
    ("sgd",      dict(optimizer_name="sgd")),
    ("adamw",    dict(optimizer_name="adamw")),
]

# ---------------------------------------------------------------------------
# HOG configuration
# ---------------------------------------------------------------------------
HOG_IMAGE_SIZES = [128, 64]
HOG_CLASSIFIERS = ["svm", "lr"]
HOG_ORIENTATIONS = 9
HOG_PPC = (8, 8)
HOG_CPB = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

# ---------------------------------------------------------------------------
# 24 missing combos (ResNet-18 & EfficientNet-B0)
# ---------------------------------------------------------------------------
REST_MODELS = ["resnet18", "efficientnet_b0"]
MISSING_COMBOS = [
    {"image_size": 64,  "augment": False, "optimizer_name": "adam"},
    {"image_size": 64,  "augment": True,  "optimizer_name": "sgd"},
    {"image_size": 64,  "augment": True,  "optimizer_name": "adamw"},
    {"image_size": 64,  "augment": False, "optimizer_name": "sgd"},
    {"image_size": 64,  "augment": False, "optimizer_name": "adamw"},
    {"image_size": 128, "augment": False, "optimizer_name": "sgd"},
    {"image_size": 128, "augment": False, "optimizer_name": "adamw"},
    {"image_size": 224, "augment": False, "optimizer_name": "adam"},
    {"image_size": 224, "augment": True,  "optimizer_name": "sgd"},
    {"image_size": 224, "augment": True,  "optimizer_name": "adamw"},
    {"image_size": 224, "augment": False, "optimizer_name": "sgd"},
    {"image_size": 224, "augment": False, "optimizer_name": "adamw"},
]

# ===== HOG evaluation helpers (from eval_hog.py) ===========================


def _extract_hog_features(csv_path: Path, image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load images from csv, extract HOG features → (X, y)."""
    df = pd.read_csv(csv_path)
    X_list = []
    y = df["label_id"].values.astype(np.int64)
    for _, row in df.iterrows():
        fpath = PROJECT_ROOT / row["filepath"]
        img = Image.open(fpath).convert("L")
        img = np.array(img.resize((image_size, image_size), Image.BILINEAR))
        feats = skimage_hog(
            img,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PPC,
            cells_per_block=HOG_CPB,
            block_norm=HOG_BLOCK_NORM,
        )
        X_list.append(feats)
    return np.array(X_list, dtype=np.float32), y


def _load_hog_model_and_scaler(clf: str, image_size: int):
    """Load HOG classifier + scaler from checkpoint pkl."""
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    base = f"hog_{clf}_img{image_size}"
    pkl_path = ckpt_dir / f"{base}_model.pkl"
    scaler_path = ckpt_dir / f"{base}_scaler.pkl"

    try:
        import joblib
    except ImportError:
        import pickle as joblib  # type: ignore[no-redef]

    obj = joblib.load(pkl_path)
    if isinstance(obj, dict):
        model = obj.get("model", obj.get("clf", obj))
        scaler = obj.get("scaler")
    else:
        model = obj
        scaler = None
    if scaler is None and scaler_path.exists():
        scaler = joblib.load(scaler_path)
    if scaler is None:
        raise FileNotFoundError(
            f"Scaler not found in {pkl_path} and {scaler_path} does not exist."
        )
    return model, scaler


def _top5_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int = 5) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))


def eval_hog_one(clf: str, image_size: int) -> dict | None:
    """Evaluate one HOG model on the test set; compute F1 and update result JSON."""
    test_csv = PROJECT_ROOT / "test.csv"
    if not test_csv.exists():
        print(f"  Skip eval hog_{clf} img{image_size}: test.csv not found")
        return None
    pkl_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"hog_{clf}_img{image_size}_model.pkl"
    if not pkl_path.exists():
        print(f"  Skip eval hog_{clf} img{image_size}: {pkl_path} not found")
        return None

    X_test, y_test = _extract_hog_features(test_csv, image_size)
    model, scaler = _load_hog_model_and_scaler(clf, image_size)
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)

    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    top5 = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)
        top5 = _top5_accuracy(y_test, probs, k=5)

    base = f"hog_{clf}_img{image_size}"
    result_path = PROJECT_ROOT / "outputs" / "logs" / f"{base}_results.json"
    out: dict = {}
    if result_path.exists():
        out = json.loads(result_path.read_text())
    out.update(
        test_accuracy=acc,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        model=f"hog_{clf}",
        image_size=image_size,
        classifier=clf,
    )
    if top5 is not None:
        out["top5_accuracy"] = top5
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(out, indent=2))
    print(f"  {base}: acc={acc:.4f}, F1_macro={f1_macro:.4f}, F1_weighted={f1_weighted:.4f}, top5={top5}")
    return out


# ===== Main entry ==========================================================


def run_all(epochs: int = 25, num_workers: int = 0, quick: bool = False):
    ensure_dirs()
    from src.train import train
    from src.eval import evaluate_model
    from src.train_hog import train_hog

    all_results: list[dict] = []
    ablations = ABLATIONS[:1] if quick else ABLATIONS

    # ---- Phase 1: CNN ablations (original 6 combos × 3 models) ----
    for model_name in CNN_MODELS:
        for abl_name, overrides in ablations:
            kwargs = dict(
                model_name=model_name,
                image_size=DEFAULT_IMG_SIZE,
                augment=True,
                optimizer_name="adam",
                num_epochs=epochs,
                num_workers=num_workers,
            )
            kwargs.update(overrides)

            if model_name == "vit_b16" and kwargs["image_size"] != VIT_REQUIRED_SIZE:
                print(f"\n  SKIP: {abl_name} (ViT requires {VIT_REQUIRED_SIZE}, got {kwargs['image_size']})")
                continue

            tag = f"{model_name}_img{kwargs['image_size']}_aug{int(kwargs['augment'])}_{kwargs['optimizer_name']}"
            print(f"\n{'#'*60}")
            print(f"  EXPERIMENT: {tag}")
            print(f"{'#'*60}")

            t0 = time.time()
            try:
                train(**kwargs)
                r = evaluate_model(
                    model_name=model_name,
                    image_size=kwargs["image_size"],
                    augment=kwargs["augment"],
                    optimizer_name=kwargs["optimizer_name"],
                    num_workers=num_workers,
                )
                r["ablation"] = abl_name
                r["elapsed_s"] = round(time.time() - t0, 1)
                all_results.append(r)
            except Exception as e:
                print(f"  *** FAILED: {e}")
                all_results.append({"model": model_name, "ablation": abl_name, "tag": tag, "error": str(e)})

    # ---- Phase 2: HOG baselines (train) ----
    for img_size in HOG_IMAGE_SIZES:
        for clf in HOG_CLASSIFIERS:
            print(f"\n{'#'*60}")
            print(f"  EXPERIMENT: HOG+{clf.upper()} / img{img_size}")
            print(f"{'#'*60}")
            t0 = time.time()
            try:
                r = train_hog(image_size=img_size, classifier=clf)
                r["ablation"] = f"hog_{clf}"
                r["elapsed_s"] = round(time.time() - t0, 1)
                all_results.append(r)
            except Exception as e:
                print(f"  *** FAILED: {e}")
                all_results.append({"model": f"hog_{clf}", "ablation": f"hog_{clf}", "error": str(e)})

    # ---- Phase 3: HOG extra evaluation (F1 macro / weighted) ----
    print(f"\n{'='*60}")
    print("  HOG EVAL: computing F1 (macro/weighted), updating result JSONs")
    print(f"{'='*60}")
    for clf in HOG_CLASSIFIERS:
        for img_size in HOG_IMAGE_SIZES:
            try:
                eval_hog_one(clf, img_size)
            except Exception as e:
                print(f"  *** eval_hog {clf} img{img_size} FAILED: {e}")

    # ---- Phase 4: 24 remaining ablation combos (ResNet-18 & EfficientNet-B0) ----
    if not quick:
        print(f"\n{'='*60}")
        print("  RUNNING 24 REMAINING ABLATION COMBOS")
        print(f"{'='*60}")
        for model_name in REST_MODELS:
            for combo in MISSING_COMBOS:
                img = combo["image_size"]
                aug = combo["augment"]
                opt = combo["optimizer_name"]
                tag = f"{model_name}_img{img}_aug{int(aug)}_{opt}"

                print(f"\n{'#'*60}")
                print(f"  EXPERIMENT: {tag}")
                print(f"{'#'*60}")

                t0 = time.time()
                try:
                    train(
                        model_name=model_name,
                        image_size=img,
                        augment=aug,
                        optimizer_name=opt,
                        num_epochs=epochs,
                        num_workers=num_workers,
                    )
                    r = evaluate_model(
                        model_name=model_name,
                        image_size=img,
                        augment=aug,
                        optimizer_name=opt,
                        num_workers=num_workers,
                    )
                    r["experiment"] = tag
                    r["elapsed_s"] = round(time.time() - t0, 1)
                    all_results.append(r)
                except Exception as e:
                    print(f"  *** FAILED: {e}")
                    all_results.append({"experiment": tag, "error": str(e)})

    # ---- Save & summarize ----
    out = PROJECT_ROOT / "outputs" / "logs" / "all_results.json"
    out.write_text(json.dumps(all_results, indent=2))

    print(f"\n{'='*60}")
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'Ablation':<30} {'Test Acc':>9}  {'Top-5':>7}  {'Time':>7}")
    print(f"  {'─'*20} {'─'*30} {'─'*9}  {'─'*7}  {'─'*7}")
    for r in all_results:
        if "error" in r:
            lbl = r.get("model", r.get("experiment", "?"))
            print(f"  {lbl:<20} {r.get('ablation', r.get('experiment', '')):<30}   FAILED")
        else:
            acc = f"{r['test_accuracy']:.4f}" if "test_accuracy" in r else "—"
            t5 = f"{r.get('top5_accuracy', 0):.4f}" if r.get("top5_accuracy") else "—"
            t = f"{r.get('elapsed_s', 0):.0f}s"
            lbl = r.get("model", r.get("experiment", ""))
            abl = r.get("ablation", r.get("experiment", ""))
            print(f"  {lbl:<20} {abl:<30} {acc:>9}  {t5:>7}  {t:>7}")
    print(f"\n  Full results: {out}")


def main():
    parser = argparse.ArgumentParser(description="Run all Caltech-101 experiments")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="Only run baseline (no ablations) for a quick smoke test")
    args = parser.parse_args()
    run_all(epochs=args.epochs, num_workers=args.num_workers, quick=args.quick)


if __name__ == "__main__":
    main()
