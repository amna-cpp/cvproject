from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

from common import DATA_DIR, FILTER_NAMES, PROJECT_ROOT, RESULTS_DIR, ensure_project_dirs, list_images


SUBSET_SIZE = 300

CONFIG = {
    "project_root": PROJECT_ROOT,
    "lfw_original_dir": DATA_DIR / "lfw_original",
    "lfw_filtered_dir": DATA_DIR / "lfw_filtered",
    "filter_names": FILTER_NAMES,
    "workers": max(1, mp.cpu_count() - 1),
    "overwrite": False,
    "jpeg_quality": 95,
    "errors_log": RESULTS_DIR / "filter_errors.log",
    "subset_size": SUBSET_SIZE,
}

try:
    import cv2
except Exception:  # pragma: no cover - dependency fallback
    cv2 = None

try:
    import mediapipe as mediapipe
except Exception:  # pragma: no cover - dependency fallback
    mediapipe = None


_FACE_MESH = None
_FACE_CASCADE = None
_EYE_CASCADE = None


def blur(image: Image.Image) -> Image.Image:
    blurred = image.filter(ImageFilter.GaussianBlur(radius=3))
    return Image.blend(image, blurred, alpha=0.4)


def brightness(image: Image.Image) -> Image.Image:
    bright = ImageEnhance.Brightness(image).enhance(1.2)
    return ImageEnhance.Color(bright).enhance(1.15)


def skin_smooth(image: Image.Image) -> Image.Image:
    if cv2 is None:
        return image.copy()
    arr = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    lower_1 = np.array([0, 30, 60], dtype=np.uint8)
    upper_1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_2 = np.array([170, 30, 60], dtype=np.uint8)
    upper_2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_1, upper_1) | cv2.inRange(hsv, lower_2, upper_2)
    mask = cv2.medianBlur(mask, 5)
    filtered = cv2.bilateralFilter(arr, d=9, sigmaColor=75, sigmaSpace=75)
    output = np.where(mask[:, :, None] > 0, filtered, arr)
    return Image.fromarray(output.astype(np.uint8), mode="RGB")


def _get_face_mesh():
    global _FACE_MESH
    if mediapipe is None or not hasattr(mediapipe, "solutions"):
        return None
    if _FACE_MESH is None:
        _FACE_MESH = mediapipe.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
        )
    return _FACE_MESH


def _get_eye_cascade():
    global _EYE_CASCADE
    if cv2 is None:
        return None
    if _EYE_CASCADE is None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"
        _EYE_CASCADE = cv2.CascadeClassifier(str(cascade_path))
    return _EYE_CASCADE


def _local_zoom(arr: np.ndarray, center: tuple[float, float], radius: float, scale: float) -> np.ndarray:
    if cv2 is None or radius <= 1:
        return arr
    height, width = arr.shape[:2]
    cx, cy = center
    x0 = max(0, int(cx - radius))
    x1 = min(width, int(cx + radius))
    y0 = max(0, int(cy - radius))
    y1 = min(height, int(cy + radius))
    if x1 <= x0 or y1 <= y0:
        return arr

    roi = arr[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = distance < radius
    falloff = np.clip(1.0 - distance / radius, 0.0, 1.0) ** 2
    local_scale = 1.0 + (scale - 1.0) * falloff
    map_x = xx.copy()
    map_y = yy.copy()
    map_x[mask] = cx + (xx[mask] - cx) / local_scale[mask]
    map_y[mask] = cy + (yy[mask] - cy) / local_scale[mask]
    map_x -= x0
    map_y -= y0
    warped = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    output = arr.copy()
    output[y0:y1, x0:x1] = warped
    return output


def _eye_enlarge_with_status(image: Image.Image) -> tuple[Image.Image, str | None]:
    if cv2 is None:
        return image.copy(), "eye_enlarge skipped: cv2 is unavailable"

    arr = np.array(image.convert("RGB"))
    face_mesh = _get_face_mesh()
    if face_mesh is None:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        eye_cascade = _get_eye_cascade()
        if eye_cascade is None or eye_cascade.empty():
            return image.copy(), "eye_enlarge skipped: eye cascade unavailable"
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(8, 8))
        if len(eyes) == 0:
            return image.copy(), "eye_enlarge skipped: no eyes detected"
        eyes = sorted(eyes, key=lambda box: box[2] * box[3], reverse=True)[:2]
        output = arr.copy()
        for x, y, w, h in eyes:
            center = (float(x + w / 2), float(y + h / 2))
            radius = max(8.0, float(max(w, h)) * 0.95)
            output = _local_zoom(output, center, radius, scale=1.08)
        return Image.fromarray(output.astype(np.uint8), mode="RGB"), "eye_enlarge used OpenCV eye cascade fallback"

    result = face_mesh.process(arr)
    if not result.multi_face_landmarks:
        return image.copy(), "eye_enlarge skipped: no landmarks detected"

    landmarks = result.multi_face_landmarks[0].landmark
    height, width = arr.shape[:2]
    eye_indices = {
        "left": [33, 133, 159, 145, 153, 154, 155, 173],
        "right": [362, 263, 386, 374, 380, 381, 382, 398],
    }

    output = arr.copy()
    for indices in eye_indices.values():
        points = np.array([(landmarks[idx].x * width, landmarks[idx].y * height) for idx in indices])
        center = points.mean(axis=0)
        radius = max(8.0, float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) * 0.85)
        output = _local_zoom(output, (float(center[0]), float(center[1])), radius, scale=1.08)
    return Image.fromarray(output.astype(np.uint8), mode="RGB"), None


def eye_enlarge(image: Image.Image) -> Image.Image:
    return _eye_enlarge_with_status(image)[0]


def _get_face_cascade():
    global _FACE_CASCADE
    if cv2 is None:
        return None
    if _FACE_CASCADE is None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(str(cascade_path))
    return _FACE_CASCADE


def _face_slim_with_status(image: Image.Image) -> tuple[Image.Image, str | None]:
    if cv2 is None:
        return image.copy(), "face_slim skipped: cv2 is unavailable"
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    cascade = _get_face_cascade()
    if cascade is None or cascade.empty():
        return image.copy(), "face_slim skipped: Haar cascade unavailable"
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return image.copy(), "face_slim skipped: no face detected"

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    roi = arr[y : y + h, x : x + w]
    new_w = max(1, int(w * 0.92))
    compressed = cv2.resize(roi, (new_w, h), interpolation=cv2.INTER_LINEAR)
    output = arr.copy()
    x_new = x + (w - new_w) // 2
    output[y : y + h, x_new : x_new + new_w] = compressed
    return Image.fromarray(output.astype(np.uint8), mode="RGB"), None


def face_slim(image: Image.Image) -> Image.Image:
    return _face_slim_with_status(image)[0]


def color_tone(image: Image.Image) -> Image.Image:
    arr = np.array(image.convert("RGB")).astype(np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + 15, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] - 10, 0, 255)

    height, width = arr.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width / 2.0, height / 2.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    distance /= max(distance.max(), 1.0)
    vignette = 1.0 - 0.15 * np.clip(distance, 0.0, 1.0)
    arr *= vignette[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


FILTER_FUNCTIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "blur": blur,
    "brightness": brightness,
    "skin_smooth": skin_smooth,
    "eye_enlarge": eye_enlarge,
    "face_slim": face_slim,
    "color_tone": color_tone,
}


def apply_filter_with_status(filter_name: str, image: Image.Image) -> tuple[Image.Image, str | None]:
    if filter_name == "eye_enlarge":
        return _eye_enlarge_with_status(image)
    if filter_name == "face_slim":
        return _face_slim_with_status(image)
    return FILTER_FUNCTIONS[filter_name](image), None


def process_one(args: tuple[str, str, str, bool, int]) -> tuple[bool, str | None]:
    image_path_str, filter_name, output_path_str, overwrite, jpeg_quality = args
    image_path = Path(image_path_str)
    output_path = Path(output_path_str)
    if output_path.exists() and not overwrite:
        return True, None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            filtered, warning = apply_filter_with_status(filter_name, image)
            filtered.save(output_path, quality=jpeg_quality)
        if warning:
            return True, f"{filter_name},{image_path},{warning}"
        return True, None
    except Exception as exc:
        return False, f"{filter_name},{image_path},{type(exc).__name__}: {exc}"


def build_jobs(filter_name: str, image_paths: list[Path]) -> list[tuple[str, str, str, bool, int]]:
    jobs = []
    source_root = CONFIG["lfw_original_dir"]
    output_root = CONFIG["lfw_filtered_dir"] / filter_name
    for image_path in image_paths:
        relative = image_path.relative_to(source_root)
        jobs.append(
            (
                str(image_path),
                filter_name,
                str(output_root / relative),
                bool(CONFIG["overwrite"]),
                int(CONFIG["jpeg_quality"]),
            )
        )
    return jobs


def subset_pair_image_paths(subset_size: int | None) -> list[Path] | None:
    if subset_size is None or subset_size <= 0:
        return None
    image_paths: set[Path] = set()
    for csv_name in ["pairs_genuine.csv", "pairs_impostor.csv"]:
        csv_path = DATA_DIR / csv_name
        if not csv_path.exists():
            continue
        pairs_df = pd.read_csv(csv_path).head(subset_size)
        for column in ["img1_path", "img2_path"]:
            for value in pairs_df[column].dropna():
                path = Path(value)
                if path.exists():
                    image_paths.add(path)
    return sorted(image_paths)


def run(
    filters: list[str] | None = None,
    workers: int | None = None,
    overwrite: bool | None = None,
    subset_size: int | None = None,
) -> dict:
    ensure_project_dirs()
    selected_filters = filters or CONFIG["filter_names"]
    if overwrite is not None:
        CONFIG["overwrite"] = overwrite
    if workers is not None:
        CONFIG["workers"] = max(1, workers)

    image_paths = subset_pair_image_paths(subset_size)
    if image_paths is None:
        image_paths = list_images(CONFIG["lfw_original_dir"])
    if not image_paths:
        raise FileNotFoundError("No LFW images found. Run src/01_setup_dataset.py first.")

    all_errors: list[str] = []
    summary: dict[str, dict] = {}

    for filter_name in selected_filters:
        if filter_name not in FILTER_FUNCTIONS:
            raise ValueError(f"Unknown filter '{filter_name}'. Valid filters: {', '.join(FILTER_FUNCTIONS)}")
        jobs = build_jobs(filter_name, image_paths)
        processed = 0
        with mp.Pool(processes=CONFIG["workers"]) as pool:
            for ok, message in tqdm(
                pool.imap_unordered(process_one, jobs),
                total=len(jobs),
                desc=f"Applying {filter_name}",
            ):
                processed += int(ok)
                if message:
                    all_errors.append(message)
        summary[filter_name] = {"processed": processed, "total": len(jobs)}

    if all_errors:
        CONFIG["errors_log"].parent.mkdir(parents=True, exist_ok=True)
        with CONFIG["errors_log"].open("a", encoding="utf-8") as handle:
            for line in all_errors:
                handle.write(line + "\n")

    print("Filter application complete")
    for filter_name, info in summary.items():
        print(f"{filter_name}: {info['processed']}/{info['total']} images")
    if all_errors:
        print(f"Logged {len(all_errors)} warnings/errors to {CONFIG['errors_log']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply simulated facial filters to LFW images.")
    parser.add_argument("--filters", default=",".join(FILTER_NAMES), help="Comma-separated filter names.")
    parser.add_argument("--workers", type=int, default=CONFIG["workers"], help="Multiprocessing workers.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing filtered images.")
    parser.add_argument("--subset", type=int, default=0, help="Pairs per class to process. Use 0 for all images.")
    args = parser.parse_args()
    filters = [part.strip() for part in args.filters.split(",") if part.strip()]
    run(filters=filters, workers=args.workers, overwrite=args.overwrite, subset_size=args.subset)


if __name__ == "__main__":
    main()
