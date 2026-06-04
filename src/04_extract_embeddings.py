from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    ALL_FILTER_NAMES,
    DATA_DIR,
    EMBEDDINGS_DIR,
    MODEL_DIMENSIONS,
    PROJECT_ROOT,
    RESULTS_DIR,
    deepface_model_name,
    ensure_project_dirs,
    image_identity,
    image_key,
    list_images,
    normalize_model_key,
    parse_models,
)


SUBSET_SIZE = 300

CONFIG = {
    "project_root": PROJECT_ROOT,
    "lfw_original_dir": DATA_DIR / "lfw_original",
    "lfw_filtered_dir": DATA_DIR / "lfw_filtered",
    "filter_names": ALL_FILTER_NAMES,
    "models": ["arcface", "facenet"],
    "embedding_index_path": EMBEDDINGS_DIR / "embedding_index.csv",
    "errors_log": RESULTS_DIR / "embedding_errors.log",
    "overwrite": False,
    "subset_size": SUBSET_SIZE,
    "detector_backend": "skip",
}


def deepface_represent(image_path: Path, model_key: str) -> np.ndarray:
    from deepface import DeepFace

    result = DeepFace.represent(
        img_path=str(image_path),
        model_name=deepface_model_name(model_key),
        enforce_detection=False,
        detector_backend=CONFIG["detector_backend"],
    )
    if isinstance(result, list):
        if not result:
            raise ValueError("DeepFace returned no embedding")
        result = result[0]
    embedding = np.asarray(result["embedding"], dtype=np.float32)
    return embedding.reshape(-1)


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
    records: list[dict] = []
    allowed_keys = subset_image_keys(subset_size)
    original_root = CONFIG["lfw_original_dir"]
    for filter_name in CONFIG["filter_names"]:
        root = original_root if filter_name == "original" else CONFIG["lfw_filtered_dir"] / filter_name
        for image_path in list_images(root):
            key = image_key(image_path)
            if allowed_keys is not None and key not in allowed_keys:
                continue
            records.append(
                {
                    "img_path": image_path,
                    "identity": image_identity(image_path),
                    "filter_name": filter_name,
                    "image_key": key,
                }
            )
    return records


def output_embedding_path(record: dict, model_key: str) -> Path:
    filename = Path(record["img_path"]).stem + ".npy"
    return EMBEDDINGS_DIR / model_key / record["filter_name"] / record["identity"] / filename


def extract_one(record: dict, model_key: str) -> tuple[dict, str | None]:
    output_path = output_embedding_path(record, model_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    is_valid = True
    error = ""

    if output_path.exists() and not CONFIG["overwrite"]:
        try:
            embedding = np.load(output_path)
            if np.any(embedding) and embedding.reshape(-1).shape[0] == MODEL_DIMENSIONS.get(model_key, 512):
                index_row = {
                    "img_path": str(record["img_path"]),
                    "identity": record["identity"],
                    "filter_name": record["filter_name"],
                    "model_name": model_key,
                    "embedding_path": str(output_path),
                    "is_valid": True,
                    "error": "",
                }
                return index_row, None
            error = "existing invalid or zero placeholder; regenerated"
        except Exception as exc:
            error = f"existing embedding unreadable; regenerated: {exc}"

    try:
        embedding = deepface_represent(Path(record["img_path"]), model_key)
        np.save(output_path, embedding.astype(np.float32))
    except Exception as exc:
        is_valid = False
        error = f"{type(exc).__name__}: {exc}"
        dim = MODEL_DIMENSIONS.get(model_key, 512)
        np.save(output_path, np.zeros(dim, dtype=np.float32))

    index_row = {
        "img_path": str(record["img_path"]),
        "identity": record["identity"],
        "filter_name": record["filter_name"],
        "model_name": model_key,
        "embedding_path": str(output_path),
        "is_valid": is_valid,
        "error": error,
    }
    log_line = None
    if not is_valid:
        log_line = f"{model_key},{record['filter_name']},{record['img_path']},{error}"
    return index_row, log_line


def run(models: list[str] | str = "arcface,facenet", overwrite: bool = False, subset_size: int | None = None) -> pd.DataFrame:
    ensure_project_dirs()
    model_keys = parse_models(models)
    CONFIG["models"] = model_keys
    CONFIG["overwrite"] = overwrite

    records = dataset_records(subset_size=subset_size)
    if not records:
        raise FileNotFoundError("No images found. Run setup and filter scripts first.")

    index_rows: list[dict] = []
    errors: list[str] = []
    for model_key in model_keys:
        for record in tqdm(records, desc=f"Extracting {model_key}"):
            row, log_line = extract_one(record, model_key)
            index_rows.append(row)
            if log_line:
                errors.append(log_line)

    index_df = pd.DataFrame(index_rows)
    CONFIG["embedding_index_path"].parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(CONFIG["embedding_index_path"], index=False)

    if errors:
        CONFIG["errors_log"].parent.mkdir(parents=True, exist_ok=True)
        with CONFIG["errors_log"].open("a", encoding="utf-8") as handle:
            for line in errors:
                handle.write(line + "\n")

    print("Embedding extraction complete")
    print(f"Total embeddings indexed: {len(index_df)}")
    for (model_name, filter_name), group in index_df.groupby(["model_name", "filter_name"]):
        failure_rate = 1.0 - float(group["is_valid"].mean())
        print(f"{model_name:>8} | {filter_name:<12} | failure rate: {failure_rate:.2%}")
    return index_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ArcFace and FaceNet embeddings with DeepFace.")
    parser.add_argument("--models", default="arcface,facenet", help="Comma-separated models: arcface,facenet")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing embeddings.")
    parser.add_argument("--subset", type=int, default=0, help="Pairs per class to process. Use 0 for all images.")
    args = parser.parse_args()
    run(models=args.models, overwrite=args.overwrite, subset_size=args.subset)


if __name__ == "__main__":
    main()
