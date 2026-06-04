from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from common import (
    ALL_FILTER_NAMES,
    DATA_DIR,
    EMBEDDINGS_DIR,
    EMBEDDING_DIM,
    METRICS_DIR,
    MODEL_DIMENSIONS,
    PLOTS_DIR,
    PROJECT_ROOT,
    compute_metrics,
    cosine_similarity,
    embedding_path_for,
    ensure_project_dirs,
    load_embedding,
    load_pairs,
    normalize_model_key,
    read_json,
    save_json,
)


SUBSET_SIZE = 300

CONFIG = {
    "project_root": PROJECT_ROOT,
    "filter_names": ALL_FILTER_NAMES,
    "model_name": "arcface",
    "subset_size": SUBSET_SIZE,
    "recognition_metrics_path": METRICS_DIR / "recognition_metrics.json",
    "pair_scores_path": METRICS_DIR / "pair_scores_arcface.csv",
    "det_prefix": METRICS_DIR / "det_curve",
}


def output_suffix(model_key: str) -> str:
    return "" if model_key == "arcface" else f"_{model_key}"


def normalize_backbone(model_name: str) -> str:
    key = model_name.strip().lower()
    if key in {"arcface", "facenet"}:
        return normalize_model_key(key)
    if key == "adaface":
        return "adaface"
    raise ValueError("Unsupported backbone. Use arcface, facenet, or adaface.")


def embedding_path_for_backbone(original_image_path: Path, filter_name: str, model_key: str) -> Path:
    if model_key == "adaface":
        return EMBEDDINGS_DIR / "adaface" / filter_name / original_image_path.parent.name / (original_image_path.stem + ".npy")
    return embedding_path_for(original_image_path, filter_name, model_key)


def score_pairs_for_filter(pairs_df: pd.DataFrame, filter_name: str, model_key: str) -> pd.DataFrame:
    expected_dim = EMBEDDING_DIM if model_key == "adaface" else MODEL_DIMENSIONS.get(model_key, 512)
    left_embeddings = []
    right_embeddings = []
    labels = []
    pair_ids = []
    for pair_id, pair in tqdm(
        pairs_df.iterrows(),
        total=len(pairs_df),
        desc=f"Scoring {filter_name}",
        leave=False,
    ):
        img1 = Path(pair["img1_path"])
        img2 = Path(pair["img2_path"])
        left_embeddings.append(load_embedding(embedding_path_for_backbone(img1, filter_name, model_key), expected_dim))
        right_embeddings.append(load_embedding(embedding_path_for_backbone(img2, filter_name, model_key), expected_dim))
        labels.append(int(pair["label"]))
        pair_ids.append(int(pair_id))
    if not left_embeddings:
        return pd.DataFrame(columns=["pair_id", "filter_name", "similarity_score", "true_label", "model_name"])
    left = np.vstack(left_embeddings)
    right = np.vstack(right_embeddings)
    numerators = np.sum(left * right, axis=1)
    denominators = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    similarities = numerators / np.clip(denominators, 1e-12, None)
    return pd.DataFrame(
        {
            "pair_id": pair_ids,
            "filter_name": filter_name,
            "similarity_score": similarities,
            "true_label": labels,
            "model_name": model_key,
        }
    )


def colors_for_filters(filters: list[str]) -> list:
    impact_scores = read_json(METRICS_DIR / "filter_impact_scores.json", default={})
    raw_values = [
        impact_scores.get(filter_name, {}).get("impact_score", 0.0)
        for filter_name in filters
        if filter_name != "original"
    ]
    min_value = min(raw_values) if raw_values else 0.0
    max_value = max(raw_values) if raw_values else 1.0
    span = max(max_value - min_value, 1e-9)
    colors = []
    for filter_name in filters:
        if filter_name == "original":
            colors.append("#2ca25f")
        else:
            value = impact_scores.get(filter_name, {}).get("impact_score", min_value)
            normalized = (value - min_value) / span
            colors.append(plt.cm.OrRd(0.35 + 0.55 * normalized))
    return colors


def plot_accuracy(metrics: dict, model_key: str) -> None:
    filters = [filter_name for filter_name in CONFIG["filter_names"] if filter_name in metrics]
    values = [metrics[filter_name]["accuracy"] for filter_name in filters]
    baseline = metrics.get("original", {}).get("accuracy", np.nan)

    plt.figure(figsize=(10, 5))
    plt.bar(filters, values, color=colors_for_filters(filters))
    if np.isfinite(baseline):
        plt.axhline(baseline, linestyle="--", color="#2ca25f", linewidth=1.5, label="Original baseline")
        plt.legend()
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Filter")
    plt.title(f"Accuracy per filter ({model_key})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = PLOTS_DIR / f"accuracy_per_filter{output_suffix(model_key)}.png"
    plt.savefig(path, dpi=200)
    plt.close()


def plot_det(det_by_filter: dict[str, pd.DataFrame], model_key: str) -> None:
    plt.figure(figsize=(8, 6))
    for idx, (filter_name, det_df) in enumerate(det_by_filter.items()):
        if det_df.empty:
            continue
        fmr = np.clip(det_df["fmr"].to_numpy(dtype=float), 1e-4, 1.0)
        fnmr = np.clip(det_df["fnmr"].to_numpy(dtype=float), 1e-4, 1.0)
        if filter_name == "original":
            plt.plot(fmr, fnmr, color="black", linewidth=2.6, label=filter_name)
        else:
            plt.plot(fmr, fnmr, color=plt.cm.tab10(idx % 10), linewidth=1.6, label=filter_name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(0.001, 1.0)
    plt.ylim(0.001, 1.0)
    plt.xlabel("False Match Rate (FMR)")
    plt.ylabel("False Non-Match Rate (FNMR)")
    plt.title(f"DET curves ({model_key})")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = PLOTS_DIR / f"det_curves{output_suffix(model_key)}.png"
    plt.savefig(path, dpi=200)
    plt.close()


def plot_similarity_heatmap(scores_df: pd.DataFrame, model_key: str) -> None:
    rows = []
    for filter_name, group in scores_df.groupby("filter_name"):
        rows.append(
            {
                "filter_name": filter_name,
                "genuine": group.loc[group["true_label"] == 1, "similarity_score"].mean(),
                "impostor": group.loc[group["true_label"] == 0, "similarity_score"].mean(),
            }
        )
    heatmap_df = pd.DataFrame(rows).set_index("filter_name").reindex(CONFIG["filter_names"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="viridis", vmin=-0.1, vmax=1.0)
    plt.xlabel("Pair type")
    plt.ylabel("Filter")
    plt.title(f"Mean cosine similarity ({model_key})")
    plt.tight_layout()
    path = PLOTS_DIR / f"similarity_heatmap{output_suffix(model_key)}.png"
    plt.savefig(path, dpi=200)
    plt.close()


def run(model_name: str = "arcface") -> dict:
    ensure_project_dirs()
    model_key = normalize_backbone(model_name)
    CONFIG["model_name"] = model_key
    suffix = output_suffix(model_key)

    pairs_df = load_pairs()
    all_scores = []
    metrics_by_filter = {}
    det_by_filter = {}

    for filter_name in CONFIG["filter_names"]:
        scores_df = score_pairs_for_filter(pairs_df, filter_name, model_key)
        metrics, det_df = compute_metrics(scores_df)
        metrics_by_filter[filter_name] = metrics
        det_by_filter[filter_name] = det_df
        all_scores.append(scores_df)
        det_path = METRICS_DIR / f"det_curve_{model_key}_{filter_name}.csv"
        det_df.to_csv(det_path, index=False)

    combined_scores = pd.concat(all_scores, ignore_index=True)
    pair_scores_path = METRICS_DIR / f"pair_scores_{model_key}.csv"
    combined_scores.to_csv(pair_scores_path, index=False)

    metrics_path = METRICS_DIR / f"recognition_metrics{suffix}.json"
    save_json(metrics_path, metrics_by_filter)
    if model_key == "arcface":
        save_json(CONFIG["recognition_metrics_path"], metrics_by_filter)

    plot_accuracy(metrics_by_filter, model_key)
    plot_det(det_by_filter, model_key)
    plot_similarity_heatmap(combined_scores, model_key)

    print(f"Recognition evaluation complete for {model_key}")
    print("Filter | Accuracy | FNMR@0.1% FMR | FNMR@1% FMR | Threshold")
    for filter_name, metrics in metrics_by_filter.items():
        print(
            f"{filter_name:<12} | {metrics['accuracy']:.2%} | "
            f"{metrics['fnmr_fmr001']:.2%} | {metrics['fnmr_fmr01']:.2%} | "
            f"{metrics['optimal_threshold']:.3f}"
        )
    return metrics_by_filter


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate verification accuracy for each filter.")
    parser.add_argument("--model", default=None, choices=["arcface", "facenet", "adaface"])
    parser.add_argument("--backbone", default=None, choices=["arcface", "facenet", "adaface"])
    args = parser.parse_args()
    run(model_name=args.backbone or args.model or "arcface")


if __name__ == "__main__":
    main()
