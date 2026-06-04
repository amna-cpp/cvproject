from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from tqdm import tqdm

from common import (
    DATA_DIR,
    EMBEDDING_DIM,
    FILTER_NAMES,
    METRICS_DIR,
    PROJECT_ROOT,
    compute_metrics,
    ensure_project_dirs,
    save_json,
)


CONFIG = {
    "EMBEDDING_DIM": EMBEDDING_DIM,
    "TRAIN_RATIO": 0.8,
    "RANDOM_SEED": 42,
    "RIDGE_ALPHA": 1e-4,
    "FILTERS": ["blur", "brightness", "skin_smooth", "eye_enlarge", "face_slim", "color_tone"],
    "ARCFACE_DIR": "results/embeddings/arcface/",
    "ADAFACE_DIR": "results/embeddings/adaface/",
    "OUTPUT_DIR": "results/mitigation_linear/",
    "PAIRS_GENUINE": "data/pairs_genuine.csv",
    "PAIRS_IMPOSTOR": "data/pairs_impostor.csv",
    "OLD_METRICS": "results/metrics/mitigation_metrics.json",
}


def project_path(value: str) -> Path:
    return PROJECT_ROOT / value


def backbone_dir(backbone: str) -> Path:
    return project_path(CONFIG["ARCFACE_DIR"] if backbone == "arcface" else CONFIG["ADAFACE_DIR"])


def embedding_path(backbone: str, filter_name: str, image_path: Path) -> Path:
    return backbone_dir(backbone) / filter_name / image_path.parent.name / (image_path.stem + ".npy")


def load_embedding(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    emb = np.load(path).astype(np.float32).reshape(-1)
    if emb.shape[0] != CONFIG["EMBEDDING_DIM"] or not np.any(emb):
        return None
    return emb


def available_identities(backbone: str, filter_name: str) -> list[str]:
    original_root = backbone_dir(backbone) / "original"
    filtered_root = backbone_dir(backbone) / filter_name
    if not original_root.exists() or not filtered_root.exists():
        return []
    original_ids = {path.name for path in original_root.iterdir() if path.is_dir()}
    filtered_ids = {path.name for path in filtered_root.iterdir() if path.is_dir()}
    return sorted(original_ids & filtered_ids)


def split_identities(identities: list[str]) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(CONFIG["RANDOM_SEED"])
    shuffled = identities[:]
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * CONFIG["TRAIN_RATIO"])
    return set(shuffled[:split_idx]), set(shuffled[split_idx:])


def paired_embeddings_for_ids(backbone: str, filter_name: str, identities: set[str]) -> tuple[np.ndarray, np.ndarray]:
    X_filtered = []
    Y_original = []
    original_root = backbone_dir(backbone) / "original"
    filtered_root = backbone_dir(backbone) / filter_name
    for identity in tqdm(sorted(identities), desc=f"Pairing {backbone}/{filter_name}", leave=False):
        for original_path in sorted((original_root / identity).glob("*.npy")):
            filtered_path = filtered_root / identity / original_path.name
            original_emb = load_embedding(original_path)
            filtered_emb = load_embedding(filtered_path)
            if original_emb is None or filtered_emb is None:
                continue
            X_filtered.append(filtered_emb)
            Y_original.append(original_emb)
    if not X_filtered:
        return (
            np.empty((0, CONFIG["EMBEDDING_DIM"]), dtype=np.float32),
            np.empty((0, CONFIG["EMBEDDING_DIM"]), dtype=np.float32),
        )
    return np.vstack(X_filtered), np.vstack(Y_original)


def corrector_path(filter_name: str, backbone: str) -> Path:
    return project_path(CONFIG["OUTPUT_DIR"]) / f"corrector_{filter_name}_{backbone}.pkl"


def train_corrector(filter_name: str, backbone: str) -> tuple[str, str, Path, np.ndarray, set[str]]:
    path = corrector_path(filter_name, backbone)
    delta_path = project_path(CONFIG["OUTPUT_DIR"]) / f"delta_{filter_name}_{backbone}.npy"
    identities = available_identities(backbone, filter_name)
    train_ids, test_ids = split_identities(identities)
    X_train, Y_train = paired_embeddings_for_ids(backbone, filter_name, train_ids)
    if X_train.size == 0:
        delta = np.zeros(CONFIG["EMBEDDING_DIM"], dtype=np.float32)
        delta_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(delta_path, delta)
        return backbone, filter_name, path, delta, test_ids

    delta = np.mean(X_train - Y_train, axis=0).astype(np.float32)
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(delta_path, delta)
    if path.exists():
        return backbone, filter_name, path, delta, test_ids

    corrector = Ridge(alpha=CONFIG["RIDGE_ALPHA"], fit_intercept=True)
    corrector.fit(X_train, Y_train)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(corrector, path)
    return backbone, filter_name, path, delta, test_ids


def load_pairs() -> pd.DataFrame:
    parts = []
    for csv_path in [project_path(CONFIG["PAIRS_GENUINE"]), project_path(CONFIG["PAIRS_IMPOSTOR"])]:
        if csv_path.exists():
            parts.append(pd.read_csv(csv_path))
    if not parts:
        return pd.DataFrame(columns=["img1_path", "img2_path", "label"])
    return pd.concat(parts, ignore_index=True)


def vectors_for_pairs(pairs_df: pd.DataFrame, backbone: str, filter_name: str, condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = []
    right = []
    labels = []
    for _, pair in tqdm(
        pairs_df.iterrows(),
        total=len(pairs_df),
        desc=f"Loading pairs {backbone}/{filter_name}/{condition}",
        leave=False,
    ):
        img1 = Path(pair["img1_path"])
        img2 = Path(pair["img2_path"])
        emb1 = load_embedding(embedding_path(backbone, condition, img1))
        emb2 = load_embedding(embedding_path(backbone, condition, img2))
        if emb1 is None or emb2 is None:
            continue
        left.append(emb1)
        right.append(emb2)
        labels.append(int(pair["label"]))
    if not left:
        empty = np.empty((0, CONFIG["EMBEDDING_DIM"]), dtype=np.float32)
        return empty, empty, np.asarray([], dtype=np.int32)
    return np.vstack(left), np.vstack(right), np.asarray(labels, dtype=np.int32)


def cosine_pairwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.asarray([], dtype=np.float32)
    numerators = np.sum(A * B, axis=1)
    denominators = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return numerators / np.clip(denominators, 1e-12, None)


def metrics_and_det_from_vectors(A: np.ndarray, B: np.ndarray, labels: np.ndarray, filter_name: str) -> tuple[dict, pd.DataFrame]:
    sims = cosine_pairwise(A, B)
    scores_df = pd.DataFrame(
        {
            "filter_name": filter_name,
            "similarity_score": sims,
            "true_label": labels,
        }
    )
    metrics, det_df = compute_metrics(scores_df)
    return metrics, det_df


def metrics_from_vectors(A: np.ndarray, B: np.ndarray, labels: np.ndarray, filter_name: str) -> dict:
    metrics, _ = metrics_and_det_from_vectors(A, B, labels, filter_name)
    return metrics


def save_det_curve(path: Path, det_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    det_df.to_csv(path, index=False, encoding="utf-8")


def evaluate_filter(backbone: str, filter_name: str, delta: np.ndarray, test_ids: set[str]) -> dict:
    pairs_df = load_pairs()
    if test_ids:
        pairs_df = pairs_df[
            pairs_df["img1_path"].map(lambda value: Path(value).parent.name in test_ids)
            & pairs_df["img2_path"].map(lambda value: Path(value).parent.name in test_ids)
        ].reset_index(drop=True)
    if pairs_df.empty:
        return empty_metrics()

    original_A, original_B, original_labels = vectors_for_pairs(pairs_df, backbone, filter_name, "original")
    filtered_A, filtered_B, filtered_labels = vectors_for_pairs(pairs_df, backbone, filter_name, filter_name)

    original_metrics, _ = metrics_and_det_from_vectors(original_A, original_B, original_labels, filter_name)
    filtered_metrics, _ = metrics_and_det_from_vectors(filtered_A, filtered_B, filtered_labels, filter_name)
    meanshift_metrics, meanshift_det = metrics_and_det_from_vectors(filtered_A - delta, filtered_B - delta, filtered_labels, filter_name)

    path = corrector_path(filter_name, backbone)
    if path.exists() and filtered_A.size:
        corrector = joblib.load(path)
        linear_A = corrector.predict(filtered_A)
        linear_B = corrector.predict(filtered_B)
    else:
        linear_A, linear_B = filtered_A, filtered_B
    linear_metrics, linear_det = metrics_and_det_from_vectors(linear_A, linear_B, filtered_labels, filter_name)
    save_det_curve(METRICS_DIR / f"det_curve_{backbone}_{filter_name}_corrected.csv", meanshift_det)
    save_det_curve(METRICS_DIR / f"det_curve_{backbone}_{filter_name}_meanshift.csv", meanshift_det)
    save_det_curve(METRICS_DIR / f"det_curve_{backbone}_{filter_name}_linear.csv", linear_det)

    fnmr_meanshift = meanshift_metrics["fnmr_fmr01"]
    fnmr_linear = linear_metrics["fnmr_fmr01"]
    if fnmr_meanshift and np.isfinite(fnmr_meanshift):
        improvement = (fnmr_meanshift - fnmr_linear) / fnmr_meanshift * 100.0
    else:
        improvement = 0.0

    return {
        "accuracy_original": original_metrics["accuracy"],
        "accuracy_filtered": filtered_metrics["accuracy"],
        "accuracy_meanshift": meanshift_metrics["accuracy"],
        "accuracy_linear": linear_metrics["accuracy"],
        "fnmr_fmr001_original": original_metrics["fnmr_fmr001"],
        "fnmr_fmr001_filtered": filtered_metrics["fnmr_fmr001"],
        "fnmr_fmr001_meanshift": meanshift_metrics["fnmr_fmr001"],
        "fnmr_fmr001_linear": linear_metrics["fnmr_fmr001"],
        "fnmr_fmr1_original": original_metrics["fnmr_fmr01"],
        "fnmr_fmr1_filtered": filtered_metrics["fnmr_fmr01"],
        "fnmr_fmr1_meanshift": meanshift_metrics["fnmr_fmr01"],
        "fnmr_fmr1_linear": linear_metrics["fnmr_fmr01"],
        "improvement_pct": float(improvement),
    }


def empty_metrics() -> dict:
    return {
        "accuracy_original": float("nan"),
        "accuracy_filtered": float("nan"),
        "accuracy_meanshift": float("nan"),
        "accuracy_linear": float("nan"),
        "fnmr_fmr001_original": float("nan"),
        "fnmr_fmr001_filtered": float("nan"),
        "fnmr_fmr001_meanshift": float("nan"),
        "fnmr_fmr001_linear": float("nan"),
        "fnmr_fmr1_original": float("nan"),
        "fnmr_fmr1_filtered": float("nan"),
        "fnmr_fmr1_meanshift": float("nan"),
        "fnmr_fmr1_linear": float("nan"),
        "improvement_pct": float("nan"),
    }


def process_filter(filter_name: str, backbone: str) -> tuple[str, dict]:
    _, _, _, delta, test_ids = train_corrector(filter_name, backbone)
    return filter_name, evaluate_filter(backbone, filter_name, delta, test_ids)


def run(backbone: str = "both") -> dict:
    ensure_project_dirs()
    output_dir = project_path(CONFIG["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    backbones = ["arcface", "adaface"] if backbone == "both" else [backbone]
    all_metrics = {}
    for backbone_name in backbones:
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(process_filter)(filter_name, backbone_name)
            for filter_name in CONFIG["FILTERS"]
        )
        all_metrics[backbone_name] = {filter_name: metrics for filter_name, metrics in results}
    save_json(output_dir / "linear_mitigation_metrics.json", all_metrics)
    print(f"Linear mitigation metrics saved to {output_dir / 'linear_mitigation_metrics.json'}")
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate per-filter Ridge embedding correctors.")
    parser.add_argument("--backbone", choices=["arcface", "adaface", "both"], default="both")
    args = parser.parse_args()
    run(backbone=args.backbone)


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
    main()
