from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import (
    ADAFACE_BACKEND,
    ALL_FILTER_NAMES,
    DATA_DIR,
    EMBEDDING_DIM,
    PROJECT_ROOT,
    RESULTS_DIR,
    ensure_project_dirs,
    image_identity,
    image_key,
    list_images,
    warn_adaface_fallback_once,
)


CONFIG = {
    "BACKBONE": ADAFACE_BACKEND,
    "MODEL_VARIANT": "ir_50",
    "EMBEDDING_DIM": EMBEDDING_DIM,
    "ENFORCE_DETECTION": False,
    "SUBSET_SIZE": 300,
    "BATCH_SIZE": 32,
    "DETECTOR_BACKEND": "skip",
    "BASE_DATA_DIR": "data/",
    "ORIGINAL_DIR": "data/lfw_original/",
    "FILTERED_DIR": "data/lfw_filtered/",
    "OUTPUT_DIR": "results/embeddings/adaface/",
    "INDEX_PATH": "results/embeddings/embedding_index_adaface.csv",
    "ERROR_LOG": "results/adaface_errors.log",
}


try:
    from adaface import AdaFaceModel

    import torch

    ADAFACE_AVAILABLE = True
except ImportError:
    AdaFaceModel = None
    torch = None
    ADAFACE_AVAILABLE = False


MODEL = None


def project_path(value: str) -> Path:
    return PROJECT_ROOT / value


def load_model():
    global MODEL
    if not ADAFACE_AVAILABLE:
        warn_adaface_fallback_once()
        return None
    if MODEL is None:
        MODEL = AdaFaceModel(architecture=CONFIG["MODEL_VARIANT"], pretrained=True)
        MODEL.eval()
    return MODEL


def subset_image_keys(subset_size: int | None) -> set[str] | None:
    if subset_size is None or subset_size <= 0:
        return None

    keys: set[str] = set()
    for csv_name in ["pairs_genuine.csv", "pairs_impostor.csv"]:
        csv_path = DATA_DIR / csv_name
        if not csv_path.exists():
            continue
        pairs_df = pd.read_csv(csv_path).head(subset_size)
        for column in ["img1_path", "img2_path"]:
            keys.update(image_key(Path(value)) for value in pairs_df[column].dropna())
    return keys or None


def dataset_records(subset_size: int | None = None) -> list[dict]:
    allowed_keys = subset_image_keys(subset_size)
    records = []
    original_root = project_path(CONFIG["ORIGINAL_DIR"])
    filtered_root = project_path(CONFIG["FILTERED_DIR"])
    for filter_name in ALL_FILTER_NAMES:
        root = original_root if filter_name == "original" else filtered_root / filter_name
        for img_path in list_images(root):
            key = image_key(img_path)
            if allowed_keys is not None and key not in allowed_keys:
                continue
            records.append(
                {
                    "img_path": img_path,
                    "identity": image_identity(img_path),
                    "filter_name": filter_name,
                    "image_key": key,
                }
            )
    return records


def output_path_for(record: dict) -> Path:
    return (
        project_path(CONFIG["OUTPUT_DIR"])
        / record["filter_name"]
        / record["identity"]
        / (Path(record["img_path"]).stem + ".npy")
    )


def load_adaface_image(img_path: Path):
    image = Image.open(img_path).convert("RGB").resize((112, 112))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def extract_adaface_batch(img_paths: list[Path]) -> list[np.ndarray]:
    model = load_model()
    arrays = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for arr in executor.map(load_adaface_image, img_paths):
            arrays.append(arr)
    batch = torch.tensor(np.stack(arrays), dtype=torch.float32)
    with torch.no_grad():
        output = model(batch)
        if isinstance(output, tuple):
            embeddings, norms = output[0], output[1]
            embeddings = embeddings / torch.clamp(norms, min=1e-12)
        else:
            embeddings = output
    return [emb.detach().cpu().numpy().astype(np.float32).reshape(-1) for emb in embeddings]


def extract_ghostfacenet_embedding(img_path: Path) -> np.ndarray:
    from deepface import DeepFace

    result = DeepFace.represent(
        img_path=str(img_path),
        model_name="GhostFaceNet",
        enforce_detection=CONFIG["ENFORCE_DETECTION"],
        detector_backend=CONFIG["DETECTOR_BACKEND"],
    )
    if isinstance(result, list):
        if not result:
            raise ValueError("DeepFace returned no embedding")
        result = result[0]
    return np.asarray(result["embedding"], dtype=np.float32).reshape(-1)


def save_embedding(path: Path, embedding: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if embedding.shape[0] != CONFIG["EMBEDDING_DIM"]:
        raise ValueError(f"Expected {CONFIG['EMBEDDING_DIM']} dims, got {embedding.shape[0]}")
    np.save(path, embedding.astype(np.float32))


def is_valid_cached_embedding(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        embedding = np.load(path).astype(np.float32).reshape(-1)
    except Exception:
        return False
    return (
        embedding.shape[0] == CONFIG["EMBEDDING_DIM"]
        and np.all(np.isfinite(embedding))
        and bool(np.any(embedding))
    )


def run(subset_size: int | None = None) -> pd.DataFrame:
    ensure_project_dirs()
    project_path(CONFIG["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    if subset_size is None:
        subset_size = CONFIG["SUBSET_SIZE"]
    CONFIG["SUBSET_SIZE"] = subset_size
    records = dataset_records(subset_size=subset_size)
    index_rows = []
    errors = []
    cached = 0
    extracted = 0
    failures = 0

    pending = []
    for record in records:
        output_path = output_path_for(record)
        row = {
            "img_path": str(record["img_path"]),
            "identity": record["identity"],
            "filter_name": record["filter_name"],
            "model_name": "adaface",
            "embedding_path": str(output_path),
        }
        index_rows.append(row)
        if is_valid_cached_embedding(output_path):
            cached += 1
        else:
            pending.append((record, output_path))

    if ADAFACE_AVAILABLE:
        load_model()
        for start in tqdm(range(0, len(pending), CONFIG["BATCH_SIZE"]), desc="Extracting AdaFace"):
            batch_items = pending[start : start + CONFIG["BATCH_SIZE"]]
            try:
                embeddings = extract_adaface_batch([item[0]["img_path"] for item in batch_items])
                for (_, output_path), embedding in zip(batch_items, embeddings):
                    save_embedding(output_path, embedding)
                    extracted += 1
            except Exception as exc:
                for record, output_path in batch_items:
                    failures += 1
                    errors.append(f"{record['filter_name']},{record['img_path']},{type(exc).__name__}: {exc}")
                    save_embedding(output_path, np.zeros(CONFIG["EMBEDDING_DIM"], dtype=np.float32))
    else:
        warn_adaface_fallback_once()
        for record, output_path in tqdm(pending, desc="Extracting GhostFaceNet"):
            try:
                embedding = extract_ghostfacenet_embedding(record["img_path"])
                save_embedding(output_path, embedding)
                extracted += 1
            except Exception as exc:
                failures += 1
                errors.append(f"{record['filter_name']},{record['img_path']},{type(exc).__name__}: {exc}")
                save_embedding(output_path, np.zeros(CONFIG["EMBEDDING_DIM"], dtype=np.float32))

    index_df = pd.DataFrame(index_rows)
    index_path = project_path(CONFIG["INDEX_PATH"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(index_path, index=False, encoding="utf-8")

    if errors:
        error_log = project_path(CONFIG["ERROR_LOG"])
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with error_log.open("a", encoding="utf-8") as handle:
            for line in errors:
                handle.write(line + "\n")

    model_used = f"AdaFace {CONFIG['MODEL_VARIANT']}" if ADAFACE_AVAILABLE else "GhostFaceNet fallback"
    print(f"Model used: {model_used}")
    print(f"Total embeddings extracted: {extracted}")
    print(f"Cached (skipped): {cached}")
    print(f"Skipped {cached} cached embeddings, extracted {extracted} new ones")
    print(f"Failures: {failures} (see {CONFIG['ERROR_LOG']})")
    print(f"Embedding dimension confirmed: {CONFIG['EMBEDDING_DIM']}")
    return index_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract AdaFace or GhostFaceNet fallback embeddings.")
    parser.add_argument("--subset", type=int, default=CONFIG["SUBSET_SIZE"], help="Pairs per class to process.")
    args = parser.parse_args()
    run(subset_size=args.subset)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    os.environ.setdefault("DEEPFACE_HOME", str(PROJECT_ROOT))
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
    main()
