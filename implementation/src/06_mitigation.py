from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from common import (
    FILTER_NAMES,
    METRICS_DIR,
    MODEL_DIMENSIONS,
    PLOTS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    compute_metrics,
    cosine_similarity,
    embedding_path_for,
    ensure_project_dirs,
    image_key,
    load_embedding,
    load_pairs,
    read_json,
    save_json,
)


SUBSET_SIZE = 300

CONFIG = {
    "project_root": PROJECT_ROOT,
    "model_name": "arcface",
    "embedding_index_path": RESULTS_DIR / "embeddings" / "embedding_index.csv",
    "detector_path": RESULTS_DIR / "mitigation_detector.pkl",
    "metrics_path": METRICS_DIR / "mitigation_metrics.json",
    "plot_path": PLOTS_DIR / "mitigation_comparison.png",
    "subset_size": SUBSET_SIZE,
    "random_seed": 42,
}


def row_image_key(path_value: str) -> str:
    path = Path(path_value)
    return f"{path.parent.name}/{path.name}"


def load_index() -> pd.DataFrame:
    path = CONFIG["embedding_index_path"]
    if not path.exists():
        raise FileNotFoundError("Embedding index missing. Run src/04_extract_embeddings.py first.")
    df = pd.read_csv(path)
    df = df[(df["model_name"] == CONFIG["model_name"]) & (df["is_valid"] == True)].copy()
    df["image_key"] = df["img_path"].map(row_image_key)
    return df


def load_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading detector features"):
        features.append(np.load(row["embedding_path"]).astype(np.float32).reshape(-1))
        labels.append(0 if row["filter_name"] == "original" else 1)
    return np.vstack(features), np.asarray(labels, dtype=np.int32)


def train_detector(df: pd.DataFrame) -> tuple[Pipeline, float]:
    X, y = load_feature_matrix(df)
    if len(np.unique(y)) < 2:
        raise ValueError("Need both original and filtered embeddings to train mitigation detector.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=CONFIG["random_seed"],
        stratify=y,
    )
    detector = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    detector.fit(X_train, y_train)
    detection_accuracy = float(accuracy_score(y_test, detector.predict(X_test)))

    CONFIG["detector_path"].parent.mkdir(parents=True, exist_ok=True)
    with CONFIG["detector_path"].open("wb") as handle:
        pickle.dump(detector, handle)
    return detector, detection_accuracy


def compute_delta_vectors(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    original_df = df[df["filter_name"] == "original"].set_index("image_key")
    train_keys, _ = train_test_split(
        original_df.index.to_list(),
        test_size=0.2,
        random_state=CONFIG["random_seed"],
    )
    train_key_set = set(train_keys)

    deltas: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for filter_name in FILTER_NAMES:
        filtered_df = df[df["filter_name"] == filter_name].set_index("image_key")
        shared_keys = sorted((set(filtered_df.index) & set(original_df.index)) & train_key_set)
        shifts = []
        for key in shared_keys:
            original_embedding = np.load(original_df.loc[key, "embedding_path"]).astype(np.float32)
            filtered_embedding = np.load(filtered_df.loc[key, "embedding_path"]).astype(np.float32)
            shifts.append(filtered_embedding.reshape(-1) - original_embedding.reshape(-1))
        if shifts:
            deltas[filter_name] = np.mean(np.vstack(shifts), axis=0).astype(np.float32)
            counts[filter_name] = len(shifts)
        else:
            deltas[filter_name] = np.zeros(MODEL_DIMENSIONS["arcface"], dtype=np.float32)
            counts[filter_name] = 0
    return deltas, counts


def score_corrected_pairs(pairs_df: pd.DataFrame, filter_name: str, delta: np.ndarray) -> pd.DataFrame:
    expected_dim = MODEL_DIMENSIONS["arcface"]
    rows = []
    for pair_id, pair in tqdm(
        pairs_df.iterrows(),
        total=len(pairs_df),
        desc=f"Corrected {filter_name}",
        leave=False,
    ):
        img1 = Path(pair["img1_path"])
        img2 = Path(pair["img2_path"])
        emb1 = load_embedding(embedding_path_for(img1, filter_name, "arcface"), expected_dim) - delta
        emb2 = load_embedding(embedding_path_for(img2, filter_name, "arcface"), expected_dim) - delta
        rows.append(
            {
                "pair_id": int(pair_id),
                "filter_name": filter_name,
                "similarity_score": cosine_similarity(emb1, emb2),
                "true_label": int(pair["label"]),
            }
        )
    return pd.DataFrame(rows)


def plot_mitigation(metrics: dict) -> None:
    filters = list(metrics["filters"].keys())
    original_values = [metrics["filters"][name]["original_accuracy"] for name in filters]
    filtered_values = [metrics["filters"][name]["filtered_accuracy"] for name in filters]
    corrected_values = [metrics["filters"][name]["corrected_accuracy"] for name in filters]

    x = np.arange(len(filters))
    width = 0.26
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, original_values, width, label="Original", color="#2ca25f")
    plt.bar(x, filtered_values, width, label="Filtered", color="#d7301f")
    plt.bar(x + width, corrected_values, width, label="Corrected", color="#3182bd")
    plt.xticks(x, filters, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Filter")
    plt.title("Recognition accuracy before and after embedding correction")
    plt.legend()
    plt.tight_layout()
    CONFIG["plot_path"].parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CONFIG["plot_path"], dpi=200)
    plt.close()


def run() -> dict:
    ensure_project_dirs()
    df = load_index()
    _, detection_accuracy = train_detector(df)
    deltas, delta_counts = compute_delta_vectors(df)

    pairs_df = load_pairs()
    recognition_metrics = read_json(METRICS_DIR / "recognition_metrics.json", default={})
    original_accuracy = recognition_metrics.get("original", {}).get("accuracy", float("nan"))

    corrected_metrics = {}
    for filter_name in FILTER_NAMES:
        scores_df = score_corrected_pairs(pairs_df, filter_name, deltas[filter_name])
        metrics, det_df = compute_metrics(scores_df)
        det_df.to_csv(METRICS_DIR / f"det_curve_arcface_{filter_name}_corrected.csv", index=False)
        corrected_metrics[filter_name] = {
            "original_accuracy": original_accuracy,
            "filtered_accuracy": recognition_metrics.get(filter_name, {}).get("accuracy", float("nan")),
            "corrected_accuracy": metrics["accuracy"],
            "corrected_fnmr_fmr001": metrics["fnmr_fmr001"],
            "corrected_fnmr_fmr01": metrics["fnmr_fmr01"],
            "corrected_threshold": metrics["optimal_threshold"],
            "delta_training_count": delta_counts[filter_name],
        }

    payload = {
        "detection_accuracy": detection_accuracy,
        "filters": corrected_metrics,
    }
    save_json(CONFIG["metrics_path"], payload)
    plot_mitigation(payload)

    print(f"Filter detector accuracy: {detection_accuracy:.2%}")
    print("Filter | Filtered accuracy | Corrected accuracy | Delta samples")
    for filter_name, metrics in corrected_metrics.items():
        print(
            f"{filter_name:<12} | {metrics['filtered_accuracy']:.2%} | "
            f"{metrics['corrected_accuracy']:.2%} | {metrics['delta_training_count']}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect filtered embeddings and apply mean-shift correction.")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
