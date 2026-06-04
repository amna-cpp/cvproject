from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from common import (
    DATA_DIR,
    FILTER_NAMES,
    METRICS_DIR,
    PLOTS_DIR,
    PROJECT_ROOT,
    cosine_distance,
    ensure_project_dirs,
    list_images,
    save_json,
)


SUBSET_SIZE = 300

CONFIG = {
    "project_root": PROJECT_ROOT,
    "lfw_original_dir": DATA_DIR / "lfw_original",
    "lfw_filtered_dir": DATA_DIR / "lfw_filtered",
    "filter_names": FILTER_NAMES,
    "sample_size": 50,
    "random_seed": 42,
    "model_name": "ArcFace",
    "subset_size": SUBSET_SIZE,
    "scores_path": METRICS_DIR / "filter_impact_scores.json",
    "plot_path": PLOTS_DIR / "filter_impact_scores.png",
}


def extract_embedding(image_path: Path, model_name: str) -> np.ndarray:
    from deepface import DeepFace

    result = DeepFace.represent(
        img_path=str(image_path),
        model_name=model_name,
        enforce_detection=False,
    )
    if isinstance(result, list):
        if not result:
            raise ValueError("DeepFace returned no embedding")
        result = result[0]
    embedding = np.asarray(result["embedding"], dtype=np.float32)
    return embedding.reshape(-1)


def sample_images(sample_size: int) -> list[Path]:
    image_paths = list_images(CONFIG["lfw_original_dir"])
    if not image_paths:
        raise FileNotFoundError("No original LFW images found. Run src/01_setup_dataset.py first.")
    rng = random.Random(CONFIG["random_seed"])
    rng.shuffle(image_paths)
    return image_paths[: min(sample_size, len(image_paths))]


def compute_filter_score(filter_name: str, images: list[Path]) -> dict:
    distances = []
    failures = 0
    filtered_root = CONFIG["lfw_filtered_dir"] / filter_name
    original_root = CONFIG["lfw_original_dir"]

    for original_path in tqdm(images, desc=f"Scoring {filter_name}"):
        filtered_path = filtered_root / original_path.relative_to(original_root)
        if not filtered_path.exists():
            failures += 1
            continue
        try:
            original_embedding = extract_embedding(original_path, CONFIG["model_name"])
            filtered_embedding = extract_embedding(filtered_path, CONFIG["model_name"])
            distances.append(cosine_distance(original_embedding, filtered_embedding))
        except Exception:
            failures += 1

    impact_score = float(np.mean(distances)) if distances else float("nan")
    return {
        "filter_name": filter_name,
        "impact_score": impact_score,
        "num_images": len(distances),
        "num_failures": failures,
    }


def plot_scores(scores: dict[str, dict]) -> None:
    ordered = sorted(scores.values(), key=lambda item: item["impact_score"], reverse=True)
    names = [item["filter_name"] for item in ordered]
    values = [item["impact_score"] for item in ordered]

    plt.figure(figsize=(9, 5))
    colors = plt.cm.OrRd(np.linspace(0.45, 0.9, len(names)))
    plt.bar(names, values, color=colors)
    plt.ylabel("Mean cosine distance from original")
    plt.xlabel("Filter")
    plt.title("Filter impact score by simulated filter")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    CONFIG["plot_path"].parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CONFIG["plot_path"], dpi=200)
    plt.close()


def run(sample_size: int = 50) -> dict:
    ensure_project_dirs()
    CONFIG["sample_size"] = sample_size
    images = sample_images(sample_size)
    scores = {
        filter_name: compute_filter_score(filter_name, images)
        for filter_name in CONFIG["filter_names"]
    }
    ranked = dict(
        sorted(
            scores.items(),
            key=lambda item: item[1]["impact_score"],
            reverse=True,
        )
    )
    save_json(CONFIG["scores_path"], ranked)
    plot_scores(ranked)

    print("Filter impact ranking")
    print("Rank | Filter | Impact score | Images | Failures")
    for rank, item in enumerate(ranked.values(), start=1):
        print(
            f"{rank:>4} | {item['filter_name']:<12} | "
            f"{item['impact_score']:.4f} | {item['num_images']:>6} | {item['num_failures']:>8}"
        )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank filters by ArcFace embedding impact.")
    parser.add_argument("--sample-size", type=int, default=CONFIG["sample_size"])
    args = parser.parse_args()
    run(sample_size=args.sample_size)


if __name__ == "__main__":
    main()
