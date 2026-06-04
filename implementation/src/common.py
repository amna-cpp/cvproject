from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

os.environ.setdefault("DEEPFACE_HOME", str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

FILTER_NAMES = [
    "blur",
    "brightness",
    "skin_smooth",
    "eye_enlarge",
    "face_slim",
    "color_tone",
]
ALL_FILTER_NAMES = ["original"] + FILTER_NAMES

MODEL_NAME_MAP = {
    "arcface": "ArcFace",
    "ArcFace": "ArcFace",
    "facenet": "Facenet",
    "FaceNet": "Facenet",
    "Facenet": "Facenet",
}

MODEL_DIMENSIONS = {
    "arcface": 512,
    "facenet": 128,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_project_dirs() -> None:
    for path in [
        DATA_DIR,
        DATA_DIR / "lfw_original",
        DATA_DIR / "lfw_filtered",
        EMBEDDINGS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    for filter_name in FILTER_NAMES:
        (DATA_DIR / "lfw_filtered" / filter_name).mkdir(parents=True, exist_ok=True)


def normalize_model_key(model_name: str) -> str:
    key = model_name.strip().lower()
    if key not in {"arcface", "facenet"}:
        raise ValueError(f"Unsupported model '{model_name}'. Use arcface or facenet.")
    return key


def deepface_model_name(model_name: str) -> str:
    key = normalize_model_key(model_name)
    return MODEL_NAME_MAP[key]


def parse_models(models: str | Iterable[str]) -> list[str]:
    if isinstance(models, str):
        parts = [part.strip() for part in models.split(",")]
    else:
        parts = [str(part).strip() for part in models]
    return [normalize_model_key(part) for part in parts if part]


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def count_identities_and_images(root: Path) -> tuple[int, int]:
    identities = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    return len(identities), len(list_images(root))


def lfw_filename(identity: str, image_num: int | str) -> str:
    return f"{identity}_{int(image_num):04d}.jpg"


def lfw_image_path(root: Path, identity: str, image_num: int | str) -> Path:
    return root / identity / lfw_filename(identity, image_num)


def image_identity(path: Path) -> str:
    return path.parent.name


def image_key(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def image_path_from_key(root: Path, key: str) -> Path:
    identity, filename = key.split("/", 1)
    return root / identity / filename


def filtered_image_path(original_image_path: Path, filter_name: str) -> Path:
    key = image_key(original_image_path)
    if filter_name == "original":
        return DATA_DIR / "lfw_original" / key
    return DATA_DIR / "lfw_filtered" / filter_name / key


def embedding_path_for(original_image_path: Path, filter_name: str, model_name: str) -> Path:
    model_key = normalize_model_key(model_name)
    key = image_key(original_image_path)
    identity, filename = key.split("/", 1)
    stem = Path(filename).stem + ".npy"
    return EMBEDDINGS_DIR / model_key / filter_name / identity / stem


def load_embedding(path: Path, expected_dim: int | None = None) -> np.ndarray:
    if not path.exists():
        dim = expected_dim or 512
        return np.zeros(dim, dtype=np.float32)
    embedding = np.load(path).astype(np.float32)
    if embedding.ndim != 1:
        embedding = embedding.reshape(-1)
    return embedding


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-12 or not math.isfinite(denom):
        return 0.0
    sim = float(np.dot(vec_a, vec_b) / denom)
    if not math.isfinite(sim):
        return 0.0
    return max(-1.0, min(1.0, sim))


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(vec_a, vec_b)


def threshold_grid() -> np.ndarray:
    return np.round(np.arange(0.0, 1.0001, 0.001), 3)


def compute_metrics(scores_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    if scores_df.empty:
        empty_det = pd.DataFrame(columns=["threshold", "fmr", "fnmr"])
        return {
            "accuracy": float("nan"),
            "optimal_threshold": float("nan"),
            "fnmr_fmr001": float("nan"),
            "threshold_fmr001": float("nan"),
            "fnmr_fmr01": float("nan"),
            "threshold_fmr01": float("nan"),
            "num_pairs": 0,
        }, empty_det

    genuine = scores_df.loc[scores_df["true_label"] == 1, "similarity_score"].to_numpy()
    impostor = scores_df.loc[scores_df["true_label"] == 0, "similarity_score"].to_numpy()
    thresholds = threshold_grid()

    det_rows = []
    best_accuracy = -1.0
    best_threshold = 0.0
    labels = scores_df["true_label"].to_numpy()
    all_scores = scores_df["similarity_score"].to_numpy()

    for threshold in thresholds:
        predicted = (all_scores >= threshold).astype(int)
        accuracy = float(np.mean(predicted == labels)) if len(labels) else float("nan")
        fmr = float(np.mean(impostor >= threshold)) if len(impostor) else float("nan")
        fnmr = float(np.mean(genuine < threshold)) if len(genuine) else float("nan")
        det_rows.append({"threshold": float(threshold), "fmr": fmr, "fnmr": fnmr})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    det_df = pd.DataFrame(det_rows)

    def fnmr_at_fmr(target_fmr: float) -> tuple[float, float]:
        if len(impostor) == 0 or len(genuine) == 0:
            return float("nan"), float("nan")
        threshold = float(np.quantile(impostor, 1.0 - target_fmr, method="higher"))
        fnmr = float(np.mean(genuine < threshold))
        return fnmr, threshold

    fnmr_001, threshold_001 = fnmr_at_fmr(0.001)
    fnmr_01, threshold_01 = fnmr_at_fmr(0.01)

    metrics = {
        "accuracy": float(best_accuracy),
        "optimal_threshold": float(best_threshold),
        "fnmr_fmr001": fnmr_001,
        "threshold_fmr001": threshold_001,
        "fnmr_fmr01": fnmr_01,
        "threshold_fmr01": threshold_01,
        "num_pairs": int(len(scores_df)),
    }
    return metrics, det_df


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return default or {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pair_csv_paths() -> tuple[Path, Path]:
    return DATA_DIR / "pairs_genuine.csv", DATA_DIR / "pairs_impostor.csv"


def load_pairs() -> pd.DataFrame:
    genuine_path, impostor_path = pair_csv_paths()
    parts = []
    if genuine_path.exists():
        parts.append(pd.read_csv(genuine_path))
    if impostor_path.exists():
        parts.append(pd.read_csv(impostor_path))
    if not parts:
        raise FileNotFoundError("Run src/01_setup_dataset.py first; pair CSVs are missing.")
    return pd.concat(parts, ignore_index=True)


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ── Upgrade: model and mitigation selection ──────────────────────────────
ADAFACE_BACKEND = "ghostfacenet"
MODEL_NAMES = ["arcface", "adaface"]
EMBEDDING_DIM = 512
MITIGATION_MODES = ["none", "meanshift", "linear"]
LINEAR_ALPHA = 1e-4
IDENTITY_SPLIT = 0.8
COMPARISON_COLORS = {
    "arcface_none": "#888780",
    "arcface_meanshift": "#D85A30",
    "adaface_none": "#378ADD",
    "adaface_linear": "#639922",
}
_ADAFACE_WARNING_PRINTED = False


def warn_adaface_fallback_once() -> None:
    global _ADAFACE_WARNING_PRINTED
    if ADAFACE_BACKEND == "ghostfacenet" and not _ADAFACE_WARNING_PRINTED:
        print("AdaFace unavailable, using GhostFaceNet fallback.")
        _ADAFACE_WARNING_PRINTED = True
