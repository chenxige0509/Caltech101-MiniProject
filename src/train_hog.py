"""
Train HOG + SVM or HOG + Logistic Regression on Caltech-101 (stratified train set).
Saves model and scaler to outputs/checkpoints; evaluates on test set and returns metrics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog as skimage_hog
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import PROJECT_ROOT

# Match run_all.py HOG params
HOG_ORIENTATIONS = 9
HOG_PPC = (8, 8)
HOG_CPB = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"


def _extract_hog(csv_path: Path, image_size: int, root: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    X_list = []
    y = df["label_id"].values.astype(np.int64)
    for _, row in df.iterrows():
        fpath = root / row["filepath"]
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


def train_hog(
    image_size: int,
    classifier: str,
) -> dict:
    """
    Extract HOG from train, fit scaler and classifier (svm or lr), save to checkpoints,
    evaluate on test set. Return dict with test_accuracy, top5_accuracy, model, etc.
    """
    root = PROJECT_ROOT
    train_csv = root / "train.csv"
    test_csv = root / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("train.csv and test.csv required. Run caltech101_data_split.ipynb first.")

    X_train, y_train = _extract_hog(train_csv, image_size, root)
    X_test, y_test = _extract_hog(test_csv, image_size, root)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if classifier == "svm":
        base = LinearSVC(max_iter=2000, dual="auto")
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=2)
        clf.fit(X_train_s, y_train)
        has_proba = True
    elif classifier == "lr":
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs", C=1.0)
        clf.fit(X_train_s, y_train)
        has_proba = True
    else:
        raise ValueError(f"classifier must be 'svm' or 'lr', got {classifier!r}")

    ckpt_dir = root / "outputs" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"hog_{classifier}_img{image_size}"
    pkl_path = ckpt_dir / f"{base_name}_model.pkl"
    scaler_path = ckpt_dir / f"{base_name}_scaler.pkl"

    try:
        import joblib
    except ImportError:
        import pickle as joblib
    joblib.dump({"model": clf, "scaler": scaler}, pkl_path)
    joblib.dump(scaler, scaler_path)

    y_pred = clf.predict(X_test_s)
    test_acc = float(np.mean(y_pred == y_test))

    top5_acc = None
    if has_proba and hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_test_s)
        topk = np.argsort(-probs, axis=1)[:, :5]
        top5_acc = float(np.mean([y_test[i] in topk[i] for i in range(len(y_test))]))

    return {
        "model": f"hog_{classifier}",
        "image_size": image_size,
        "classifier": classifier,
        "test_accuracy": test_acc,
        "top5_accuracy": top5_acc,
    }

