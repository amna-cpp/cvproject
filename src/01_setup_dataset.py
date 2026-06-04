from __future__ import annotations

import argparse
import csv
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from common import (
    DATA_DIR,
    PROJECT_ROOT,
    count_identities_and_images,
    ensure_project_dirs,
    lfw_image_path,
    pair_csv_paths,
)


SUBSET_SIZE = 300

CONFIG = {
    "project_root": PROJECT_ROOT,
    "downloads_dir": Path.home() / "Downloads",
    "lfw_original_dir": DATA_DIR / "lfw_original",
    "pairs_txt_path": DATA_DIR / "pairs.txt",
    "pairs_url": "http://vis-www.cs.umass.edu/lfw/pairs.txt",
    "subset_size": SUBSET_SIZE,
    "copy_images": True,
    "local_archive_names": [
        "lfw-deepfunneled",
        "lfw",
        "archive/lfw-deepfunneled/lfw-deepfunneled",
        "archive/lfw-deepfunneled",
    ],
}


def has_lfw_images(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(candidate.suffix.lower() in {".jpg", ".jpeg", ".png"} for candidate in path.rglob("*"))


def find_nested_lfw_dir(path: Path) -> Path | None:
    if not path.exists():
        return None
    if has_lfw_images(path) and any(child.is_dir() for child in path.iterdir()):
        image_parents = {candidate.parent for candidate in path.rglob("*.jpg")}
        if any(parent.parent == path for parent in image_parents):
            return path
    for child in path.iterdir() if path.is_dir() else []:
        if child.is_dir():
            nested = find_nested_lfw_dir(child)
            if nested is not None:
                return nested
    return None


def find_lfw_source(downloads_dir: Path) -> Path:
    for name in CONFIG["local_archive_names"]:
        candidate = downloads_dir / name
        nested = find_nested_lfw_dir(candidate)
        if nested is not None:
            return nested

    archive_patterns = ["*lfw*.tgz", "*lfw*.tar.gz", "*lfw*.zip"]
    for pattern in archive_patterns:
        for archive_path in downloads_dir.glob(pattern):
            extracted = extract_archive_to_temp(archive_path)
            nested = find_nested_lfw_dir(extracted)
            if nested is not None:
                return nested

    raise FileNotFoundError(
        f"Could not find LFW images under {downloads_dir}. Expected an LFW folder or archive."
    )


def extract_archive_to_temp(archive_path: Path) -> Path:
    target = DATA_DIR / "_archive_extract" / archive_path.stem.replace(".tar", "")
    if target.exists() and has_lfw_images(target):
        return target
    target.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as handle:
            handle.extractall(target)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as handle:
            handle.extractall(target)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    return target


def prepare_lfw_images(copy_images: bool = True) -> Path:
    ensure_project_dirs()
    destination = CONFIG["lfw_original_dir"]
    if has_lfw_images(destination):
        return destination

    source = find_lfw_source(CONFIG["downloads_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        if destination.exists():
            destination.rmdir()
        destination.symlink_to(source, target_is_directory=True)
    return destination


def find_local_pairs_txt(downloads_dir: Path) -> Path | None:
    candidates = [
        downloads_dir / "pairs.txt",
        downloads_dir / "archive" / "pairs.txt",
        DATA_DIR / "pairs.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in downloads_dir.glob("**/pairs.txt"):
        return candidate
    return None


def ensure_pairs_txt() -> Path | None:
    target = CONFIG["pairs_txt_path"]
    if target.exists():
        return target
    local = find_local_pairs_txt(CONFIG["downloads_dir"])
    if local is not None:
        shutil.copy2(local, target)
        return target
    try:
        with urllib.request.urlopen(CONFIG["pairs_url"], timeout=20) as response:
            target.write_bytes(response.read())
        return target
    except Exception as exc:
        print(f"Could not download standard pairs.txt ({exc}). Falling back to local CSV pairs.")
        return None


def has_official_pairs_header(pairs_txt: Path) -> bool:
    try:
        first_line = pairs_txt.read_text(encoding="utf-8").splitlines()[0].split()
    except Exception:
        return False
    return len(first_line) >= 2


def parse_standard_pairs(pairs_txt: Path, lfw_root: Path) -> tuple[list[dict], list[dict]]:
    genuine_pairs: list[dict] = []
    impostor_pairs: list[dict] = []
    with pairs_txt.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 3:
            name, num1, num2 = parts
            img1 = lfw_image_path(lfw_root, name, num1)
            img2 = lfw_image_path(lfw_root, name, num2)
            if img1.exists() and img2.exists():
                genuine_pairs.append({"img1_path": str(img1), "img2_path": str(img2), "label": 1})
        elif len(parts) == 4:
            name1, num1, name2, num2 = parts
            img1 = lfw_image_path(lfw_root, name1, num1)
            img2 = lfw_image_path(lfw_root, name2, num2)
            if img1.exists() and img2.exists():
                impostor_pairs.append({"img1_path": str(img1), "img2_path": str(img2), "label": 0})
    return genuine_pairs, impostor_pairs


def safe_lfw_image_path(lfw_root: Path, identity: str, image_num: str) -> Path | None:
    try:
        if not identity or not str(image_num).strip():
            return None
        return lfw_image_path(lfw_root, identity, image_num)
    except ValueError:
        return None


def parse_local_csv_pairs(lfw_root: Path) -> tuple[list[dict], list[dict]]:
    downloads = CONFIG["downloads_dir"]
    mixed_pairs_file = downloads / "archive" / "pairs.csv"
    genuine_files = [
        downloads / "archive" / "matchpairsDevTrain.csv",
        downloads / "archive" / "matchpairsDevTest.csv",
    ]
    impostor_files = [
        downloads / "archive" / "mismatchpairsDevTrain.csv",
        downloads / "archive" / "mismatchpairsDevTest.csv",
    ]

    genuine_pairs: list[dict] = []
    impostor_pairs: list[dict] = []

    if mixed_pairs_file.exists():
        with mixed_pairs_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) == 3 or (len(row) >= 4 and not row[3].strip()):
                    name, num1, num2 = row[0], row[1], row[2]
                    img1 = safe_lfw_image_path(lfw_root, name, num1)
                    img2 = safe_lfw_image_path(lfw_root, name, num2)
                    if img1 is not None and img2 is not None and img1.exists() and img2.exists():
                        genuine_pairs.append({"img1_path": str(img1), "img2_path": str(img2), "label": 1})
                elif len(row) >= 4:
                    name1, num1, name2, num2 = row[0], row[1], row[2], row[3]
                    img1 = safe_lfw_image_path(lfw_root, name1, num1)
                    img2 = safe_lfw_image_path(lfw_root, name2, num2)
                    if img1 is not None and img2 is not None and img1.exists() and img2.exists():
                        impostor_pairs.append({"img1_path": str(img1), "img2_path": str(img2), "label": 0})
        if genuine_pairs and impostor_pairs:
            return genuine_pairs, impostor_pairs

    for csv_path in genuine_files:
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                name, num1, num2 = row[0], row[1], row[2]
                img1 = safe_lfw_image_path(lfw_root, name, num1)
                img2 = safe_lfw_image_path(lfw_root, name, num2)
                if img1 is not None and img2 is not None and img1.exists() and img2.exists():
                    genuine_pairs.append({"img1_path": str(img1), "img2_path": str(img2), "label": 1})

    for csv_path in impostor_files:
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                name1, num1, name2, num2 = row[0], row[1], row[2], row[3]
                img1 = safe_lfw_image_path(lfw_root, name1, num1)
                img2 = safe_lfw_image_path(lfw_root, name2, num2)
                if img1 is not None and img2 is not None and img1.exists() and img2.exists():
                    impostor_pairs.append({"img1_path": str(img1), "img2_path": str(img2), "label": 0})

    return genuine_pairs, impostor_pairs


def apply_subset(pairs: list[dict], subset_size: int | None) -> list[dict]:
    if subset_size is None or subset_size <= 0:
        return pairs
    return pairs[:subset_size]


def save_pair_csvs(genuine_pairs: list[dict], impostor_pairs: list[dict]) -> None:
    genuine_path, impostor_path = pair_csv_paths()
    pd.DataFrame(genuine_pairs).to_csv(genuine_path, index=False)
    pd.DataFrame(impostor_pairs).to_csv(impostor_path, index=False)


def image_num_from_path(path: Path) -> str:
    return str(int(path.stem.rsplit("_", 1)[-1]))


def save_pairs_txt_fallback(
    genuine_pairs: list[dict],
    impostor_pairs: list[dict],
    overwrite: bool = False,
) -> None:
    target = CONFIG["pairs_txt_path"]
    if target.exists() and not overwrite:
        return
    with target.open("w", encoding="utf-8") as handle:
        handle.write(f"{min(len(genuine_pairs), len(impostor_pairs))}\n")
        for pair in genuine_pairs:
            img1 = Path(pair["img1_path"])
            img2 = Path(pair["img2_path"])
            handle.write(f"{img1.parent.name}\t{image_num_from_path(img1)}\t{image_num_from_path(img2)}\n")
        for pair in impostor_pairs:
            img1 = Path(pair["img1_path"])
            img2 = Path(pair["img2_path"])
            handle.write(
                f"{img1.parent.name}\t{image_num_from_path(img1)}\t"
                f"{img2.parent.name}\t{image_num_from_path(img2)}\n"
            )


def run(subset_size: int = SUBSET_SIZE, copy_images: bool = True) -> dict:
    CONFIG["subset_size"] = subset_size
    CONFIG["copy_images"] = copy_images
    lfw_root = prepare_lfw_images(copy_images=copy_images)

    pairs_txt = ensure_pairs_txt()
    if pairs_txt is not None and has_official_pairs_header(pairs_txt):
        genuine_pairs, impostor_pairs = parse_standard_pairs(pairs_txt, lfw_root)
    else:
        genuine_pairs, impostor_pairs = parse_local_csv_pairs(lfw_root)
        if genuine_pairs and impostor_pairs:
            save_pairs_txt_fallback(genuine_pairs, impostor_pairs, overwrite=True)

    if not genuine_pairs or not impostor_pairs:
        print("Standard pairs were unavailable or incomplete; trying local CSV pairs.")
        genuine_pairs, impostor_pairs = parse_local_csv_pairs(lfw_root)

    genuine_pairs = apply_subset(genuine_pairs, subset_size)
    impostor_pairs = apply_subset(impostor_pairs, subset_size)
    save_pair_csvs(genuine_pairs, impostor_pairs)
    save_pairs_txt_fallback(genuine_pairs, impostor_pairs)

    identities, images = count_identities_and_images(lfw_root)
    summary = {
        "lfw_root": str(lfw_root),
        "num_identities": identities,
        "num_images": images,
        "num_genuine_pairs": len(genuine_pairs),
        "num_impostor_pairs": len(impostor_pairs),
    }

    print("Dataset setup complete")
    print(f"Identities: {identities}")
    print(f"Images: {images}")
    print(f"Genuine pairs: {len(genuine_pairs)}")
    print(f"Impostor pairs: {len(impostor_pairs)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LFW images and verification pairs.")
    parser.add_argument("--subset", type=int, default=SUBSET_SIZE, help="Pairs per class to keep.")
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink LFW into data/lfw_original instead of copying images.",
    )
    args = parser.parse_args()
    run(subset_size=args.subset, copy_images=not args.symlink)


if __name__ == "__main__":
    main()
