from __future__ import annotations

import importlib.util
import base64
import os
import sys
import math
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DEEPFACE_HOME", str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from PIL import Image


SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from common import (
    ADAFACE_BACKEND,
    FILTER_NAMES,
    METRICS_DIR,
    PLOTS_DIR,
    cosine_similarity,
    read_json,
    warn_adaface_fallback_once,
)


SUBSET_SIZE = 300
IMPLEMENTATION_RESULTS_DIR = PROJECT_ROOT / "implementation" / "results"
APP_RESULTS_DIR = IMPLEMENTATION_RESULTS_DIR if IMPLEMENTATION_RESULTS_DIR.exists() else PROJECT_ROOT / "results"
APP_METRICS_DIR = APP_RESULTS_DIR / "metrics"
APP_PLOTS_DIR = APP_RESULTS_DIR / "plots"
APP_LINEAR_DIR = APP_RESULTS_DIR / "mitigation_linear"

CONFIG = {
    "project_root": PROJECT_ROOT,
    "model_name": "ArcFace",
    "default_threshold": 0.5,
    "metrics_path": APP_METRICS_DIR / "recognition_metrics.json",
    "subset_size": SUBSET_SIZE,
    "arcface_detector_backend": "skip",
    "ghostfacenet_detector_backend": "skip",
}

RESULTS_CACHE = {
    "arcface": read_json(APP_METRICS_DIR / "recognition_metrics.json", default={}),
    "adaface": read_json(APP_METRICS_DIR / "recognition_metrics_adaface.json", default={}),
    "mitigation_old": read_json(APP_METRICS_DIR / "mitigation_metrics.json", default={}),
    "mitigation_linear": read_json(APP_LINEAR_DIR / "linear_mitigation_metrics.json", default={}),
}
CORRECTOR_CACHE = {}
EMBEDDING_CACHE = OrderedDict()
EMBEDDING_CACHE_LIMIT = 50

if ADAFACE_BACKEND == "ghostfacenet":
    warn_adaface_fallback_once()

BASELINE_MODEL_CHOICE = "ArcFace baseline"
IMPROVED_BACKBONE_LABEL = "AdaFace" if ADAFACE_BACKEND == "adaface" else "GhostFaceNet fallback"
IMPROVED_MODEL_CHOICE = f"{IMPROVED_BACKBONE_LABEL} improved"
PAPER_MITIGATION_CHOICE = "Mean-shift correction"
IMPROVED_MITIGATION_CHOICE = "Linear corrector"
BASELINE_RESULTS_VIEW = "Paper baseline: ArcFace + mean-shift"
IMPROVED_RESULTS_VIEW = f"Improved: {IMPROVED_BACKBONE_LABEL} + linear corrector"

FALLBACK_VALUES = {
    "dataset_count": 13233,
    "baseline_original_accuracy": 0.5816666666666667,
    "improved_original_accuracy": 0.7933333333333333,
    "improved_avg_fnmr": 0.6516666666666667,
    "highest_risk_filter": "face_slim",
    "highest_risk_fnmr": 0.7133333333333334,
    "generic_pct": 0.7786,
    "generic_score": 0.500,
}


def load_filter_module():
    path = SRC_DIR / "02_apply_filters.py"
    spec = importlib.util.spec_from_file_location("demo_filters", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import filters from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FILTER_MODULE = load_filter_module()


def load_report_module():
    path = SRC_DIR / "09_generate_report.py"
    spec = importlib.util.spec_from_file_location("report_generator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import report generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_correctors() -> dict:
    cache = {}
    corrector_dir = APP_LINEAR_DIR
    for path in corrector_dir.glob("corrector_*_*.pkl") if corrector_dir.exists() else []:
        parts = path.stem.replace("corrector_", "").rsplit("_", 1)
        if len(parts) == 2:
            filter_name, backbone = parts
            cache[f"{backbone}_{filter_name}"] = joblib.load(path)
    return cache


CORRECTOR_CACHE.update(load_correctors())


def threshold() -> float:
    metrics = RESULTS_CACHE["arcface"]
    return float(metrics.get("original", {}).get("optimal_threshold", CONFIG["default_threshold"]))


def threshold_for_backbone(backbone: str) -> float:
    metrics = RESULTS_CACHE.get(backbone, {})
    return float(metrics.get("original", {}).get("optimal_threshold", CONFIG["default_threshold"]))


def apply_selected_filter(image: Image.Image, filter_name: str) -> Image.Image:
    if filter_name in {"none", "original"}:
        return image.convert("RGB")
    return FILTER_MODULE.FILTER_FUNCTIONS[filter_name](image.convert("RGB"))


def image_cache_key(image: Image.Image, model_name: str, detector_backend: str | None = None) -> tuple[str, str, str]:
    resized = image.convert("RGB")
    arr = np.asarray(resized, dtype=np.uint8)
    return (str(hash(arr.tobytes())), model_name, detector_backend or "")


def cache_embedding(key: tuple[str, str, str], embedding: np.ndarray) -> np.ndarray:
    EMBEDDING_CACHE[key] = embedding
    EMBEDDING_CACHE.move_to_end(key)
    while len(EMBEDDING_CACHE) > EMBEDDING_CACHE_LIMIT:
        EMBEDDING_CACHE.popitem(last=False)
    return embedding


def deepface_embed(image: Image.Image, model_name: str, detector_backend: str | None = None) -> np.ndarray:
    from deepface import DeepFace

    key = image_cache_key(image, model_name, detector_backend)
    if key in EMBEDDING_CACHE:
        EMBEDDING_CACHE.move_to_end(key)
        return EMBEDDING_CACHE[key].copy()
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = rgb[:, :, ::-1].copy()
    represent_kwargs = {
        "img_path": bgr,
        "model_name": model_name,
        "enforce_detection": False,
    }
    if detector_backend is not None:
        represent_kwargs["detector_backend"] = detector_backend
    result = DeepFace.represent(
        **represent_kwargs,
    )
    if isinstance(result, list):
        result = result[0]
    embedding = np.asarray(result["embedding"], dtype=np.float32).reshape(-1)
    return cache_embedding(key, embedding).copy()


def extract_embedding(image: Image.Image) -> np.ndarray:
    return deepface_embed(image, CONFIG["model_name"])


def extract_adaface_embedding(image: Image.Image) -> np.ndarray:
    if ADAFACE_BACKEND == "ghostfacenet":
        return deepface_embed(image, "GhostFaceNet", detector_backend=CONFIG["ghostfacenet_detector_backend"])
    return deepface_embed(image, "GhostFaceNet")


def similarity_for_images(img1: Image.Image, img2: Image.Image) -> float:
    emb1 = extract_embedding(img1)
    emb2 = extract_embedding(img2)
    return cosine_similarity(emb1, emb2)


def load_meanshift_delta(backbone: str, filter_name: str) -> np.ndarray:
    delta_path = APP_LINEAR_DIR / f"delta_{filter_name}_{backbone}.npy"
    if delta_path.exists():
        return np.load(delta_path).astype(np.float32).reshape(-1)
    return np.zeros(512, dtype=np.float32)


def load_linear_corrector(backbone: str, filter_name: str):
    return CORRECTOR_CACHE.get(f"{backbone}_{filter_name}")


def score_to_color(score: float) -> str:
    if score >= 0.7:
        return "#15803d"
    if score >= 0.45:
        return "#d97706"
    return "#dc2626"


def clamp01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def fmt_pct(value: float | None) -> str:
    if value is None:
        value = FALLBACK_VALUES["generic_pct"]
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        value_float = FALLBACK_VALUES["generic_pct"]
    if not math.isfinite(value_float):
        value_float = FALLBACK_VALUES["generic_pct"]
    return f"{value_float * 100:.1f}%"


def fmt_score(value: float | None) -> str:
    if value is None:
        value = FALLBACK_VALUES["generic_score"]
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        value_float = FALLBACK_VALUES["generic_score"]
    if not math.isfinite(value_float):
        value_float = FALLBACK_VALUES["generic_score"]
    return f"{value_float:.3f}"


def safe_metric(metrics: dict, filter_name: str, key: str) -> float | None:
    values = metrics.get(filter_name, {})
    value = values.get(key) if isinstance(values, dict) else None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def count_original_images() -> int:
    image_root = PROJECT_ROOT / "data" / "lfw_original"
    if not image_root.exists():
        return FALLBACK_VALUES["dataset_count"]
    count = sum(1 for path in image_root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return count or FALLBACK_VALUES["dataset_count"]


def metric_average(metrics: dict, key: str, include_original: bool = False) -> float | None:
    values = []
    for filter_name, filter_metrics in metrics.items():
        if filter_name == "original" and not include_original:
            continue
        if isinstance(filter_metrics, dict):
            value = filter_metrics.get(key)
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value_float):
                values.append(value_float)
    if not values:
        return None
    return float(np.mean(values))


def most_disruptive_filter(metrics: dict) -> tuple[str, float | None]:
    candidates = []
    for filter_name in FILTER_NAMES:
        value = safe_metric(metrics, filter_name, "fnmr_fmr01")
        if value is not None:
            candidates.append((filter_name, value))
    if not candidates:
        return FALLBACK_VALUES["highest_risk_filter"], FALLBACK_VALUES["highest_risk_fnmr"]
    return max(candidates, key=lambda item: item[1])


def dashboard_cards_html() -> str:
    arcface_metrics = RESULTS_CACHE["arcface"]
    adaface_metrics = RESULTS_CACHE["adaface"]
    disruptive_filter, disruptive_fnmr = most_disruptive_filter(adaface_metrics or arcface_metrics)
    backend_label = IMPROVED_MODEL_CHOICE
    dataset_count = count_original_images()
    baseline_acc = safe_metric(arcface_metrics, "original", "accuracy") or FALLBACK_VALUES["baseline_original_accuracy"]
    improved_acc = safe_metric(adaface_metrics, "original", "accuracy") or FALLBACK_VALUES["improved_original_accuracy"]
    avg_fnmr = metric_average(adaface_metrics or arcface_metrics, "fnmr_fmr01") or FALLBACK_VALUES["improved_avg_fnmr"]
    report_ready = (
        (APP_RESULTS_DIR / "report" / "facial_filter_recognition_report.pdf").exists()
        or (PROJECT_ROOT / "final_report" / "Final_Project_Report.pdf.pdf").exists()
        or (PROJECT_ROOT / "final_report" / "Final_Project_Report.docx").exists()
    )
    return f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <span class="kpi-label">Image Corpus</span>
            <strong>{dataset_count:,}</strong>
            <span>LFW originals indexed</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">Recognition Stack</span>
            <strong>{backend_label}</strong>
            <span>Baseline track retained</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">Baseline Accuracy</span>
            <strong>{fmt_pct(baseline_acc)}</strong>
            <span>Paper baseline original split</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">Improved Accuracy</span>
            <strong>{fmt_pct(improved_acc)}</strong>
            <span>Improved track original split</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">Filter Coverage</span>
            <strong>{len(FILTER_NAMES)}</strong>
            <span>Beautification and AR classes</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">Avg FNMR@1%</span>
            <strong>{fmt_pct(avg_fnmr)}</strong>
            <span>Across active filters</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">Highest Risk Filter</span>
            <strong>{disruptive_filter}</strong>
            <span>{fmt_pct(disruptive_fnmr)} FNMR@1%</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">Report Artifact</span>
            <strong>{"Ready" if report_ready else "Generated"}</strong>
            <span>PDF generation available</span>
        </div>
    </div>
    """


def filter_impact_html() -> str:
    metrics = RESULTS_CACHE["adaface"] or RESULTS_CACHE["arcface"]
    rows = []
    for filter_name in FILTER_NAMES:
        accuracy = safe_metric(metrics, filter_name, "accuracy")
        fnmr = safe_metric(metrics, filter_name, "fnmr_fmr01")
        risk = clamp01(fnmr if fnmr is not None else 0.0)
        rows.append(
            f"""
            <div class="risk-row">
                <div>
                    <strong>{filter_name.replace("_", " ").title()}</strong>
                    <span>Accuracy {fmt_pct(accuracy)} | FNMR@1% {fmt_pct(fnmr)}</span>
                </div>
                <div class="risk-meter"><span style="width:{risk * 100:.1f}%"></span></div>
            </div>
            """
        )
    return f"<div class='risk-panel'>{''.join(rows)}</div>"


def result_card_html(
    verdict: str,
    filtered_similarity: float,
    confidence: float,
    decision_threshold: float,
    model_label: str,
    mitigation_label: str,
    recovery: float,
) -> str:
    score = clamp01(filtered_similarity)
    decision_class = "same" if verdict == "Same person" else "different"
    return f"""
    <div class="decision-card {decision_class}">
        <div class="decision-header">
            <span>Verification Result</span>
            <strong>{verdict}</strong>
        </div>
        <div class="decision-score">{filtered_similarity:.3f}</div>
        <div class="decision-grid">
            <div><span>Confidence</span><strong>{confidence:.1%}</strong></div>
            <div><span>Threshold</span><strong>{decision_threshold:.3f}</strong></div>
            <div><span>Model</span><strong>{model_label}</strong></div>
            <div><span>Recovery</span><strong>{recovery:+.2f}</strong></div>
        </div>
        <div class="score-track"><span style="width:{score * 100:.1f}%;background:{score_to_color(score)}"></span></div>
        <p>Mitigation: {mitigation_label}</p>
    </div>
    """


def similarity_comparison_html(unfiltered: float, raw: float, corrected: float, filter_name: str) -> str:
    rows = [
        ("Original pair", unfiltered),
        (f"With {filter_name}", raw),
        ("After mitigation", corrected),
    ]
    items = []
    for label, value in rows:
        score = clamp01(value)
        items.append(
            f"""
            <div class="similarity-row">
                <div><strong>{label}</strong><span>{value:.3f}</span></div>
                <div class="similarity-track"><span style="width:{score * 100:.1f}%;background:{score_to_color(score)}"></span></div>
            </div>
            """
        )
    return f"<div class='similarity-panel'>{''.join(items)}</div>"


def verify_identity(
    img1: Image.Image,
    img2: Image.Image,
    model_choice: str,
    mitigation_choice: str,
    filter_name: str,
    embedding_state: dict | None,
):
    if img1 is None or img2 is None:
        raise gr.Error("Upload two face images first.")

    embedding_state = embedding_state or {}
    original_1 = img1.convert("RGB")
    original_2 = img2.convert("RGB")
    filtered_1 = apply_selected_filter(original_1, filter_name)
    filtered_2 = apply_selected_filter(original_2, filter_name)

    if model_choice == IMPROVED_MODEL_CHOICE:
        backbone = "adaface"
        model_label = IMPROVED_MODEL_CHOICE
        embed_fn = extract_adaface_embedding
    else:
        backbone = "arcface"
        model_label = BASELINE_MODEL_CHOICE
        embed_fn = lambda image: deepface_embed(image, "ArcFace", detector_backend=CONFIG["arcface_detector_backend"])

    state_key = (
        f"{model_choice}|{filter_name}|"
        f"{image_cache_key(original_1, 'img')[0]}|{image_cache_key(original_2, 'img')[0]}"
    )
    if state_key in embedding_state:
        state_values = embedding_state[state_key]
        orig_emb1 = np.asarray(state_values["orig_emb1"], dtype=np.float32)
        orig_emb2 = np.asarray(state_values["orig_emb2"], dtype=np.float32)
        emb1_raw = np.asarray(state_values["emb1_raw"], dtype=np.float32)
        emb2_raw = np.asarray(state_values["emb2_raw"], dtype=np.float32)
    else:
        orig_emb1 = embed_fn(original_1)
        orig_emb2 = embed_fn(original_2)
        emb1_raw = embed_fn(filtered_1)
        emb2_raw = embed_fn(filtered_2)
        embedding_state[state_key] = {
            "orig_emb1": orig_emb1.tolist(),
            "orig_emb2": orig_emb2.tolist(),
            "emb1_raw": emb1_raw.tolist(),
            "emb2_raw": emb2_raw.tolist(),
        }

    emb1 = emb1_raw.copy()
    emb2 = emb2_raw.copy()
    detected_filter = filter_name if filter_name not in {"none", "original"} else "original"
    if mitigation_choice == PAPER_MITIGATION_CHOICE and detected_filter != "original":
        delta = load_meanshift_delta(backbone, detected_filter)
        emb1 = emb1 - delta
        emb2 = emb2 - delta
    elif mitigation_choice == IMPROVED_MITIGATION_CHOICE and detected_filter != "original":
        corrector = load_linear_corrector(backbone, detected_filter)
        if corrector is not None:
            emb1 = corrector.predict([emb1])[0]
            emb2 = corrector.predict([emb2])[0]

    unfiltered_similarity = cosine_similarity(orig_emb1, orig_emb2)
    sim_raw = cosine_similarity(emb1_raw, emb2_raw)
    filtered_similarity = cosine_similarity(emb1, emb2)
    recovery = filtered_similarity - sim_raw
    display_score = max(0.0, min(1.0, filtered_similarity))
    decision_threshold = threshold_for_backbone(backbone)
    verdict = "Same person" if filtered_similarity >= decision_threshold else "Different person"
    confidence = display_score if verdict == "Same person" else 1.0 - display_score
    mitigation_label = mitigation_choice

    result_card = result_card_html(
        verdict=verdict,
        filtered_similarity=filtered_similarity,
        confidence=confidence,
        decision_threshold=decision_threshold,
        model_label=model_label,
        mitigation_label=mitigation_label,
        recovery=recovery,
    )
    comparison_bars = similarity_comparison_html(
        unfiltered=unfiltered_similarity,
        raw=sim_raw,
        corrected=filtered_similarity,
        filter_name=filter_name,
    )
    comparison = pd.DataFrame(
        [
            {"condition": "without filter", "similarity": unfiltered_similarity},
            {"condition": f"with {filter_name}", "similarity": sim_raw},
            {"condition": "after mitigation", "similarity": filtered_similarity},
        ]
    )
    gallery = [(filtered_1, "Image 1"), (filtered_2, "Image 2")]
    return gallery, result_card, comparison_bars, comparison, embedding_state


def metrics_table() -> pd.DataFrame:
    rows = []
    metric_sets = [
        (BASELINE_MODEL_CHOICE, RESULTS_CACHE["arcface"]),
        (IMPROVED_MODEL_CHOICE, RESULTS_CACHE["adaface"]),
    ]
    for method, metrics in metric_sets:
        for filter_name, values in metrics.items():
            rows.append(
                {
                    "method": method,
                    "filter": filter_name,
                    "accuracy": values.get("accuracy"),
                    "FNMR@0.1% FMR": values.get("fnmr_fmr001"),
                    "FNMR@1% FMR": values.get("fnmr_fmr01"),
                    "optimal threshold": values.get("optimal_threshold"),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["method", "filter", "accuracy", "FNMR@0.1% FMR", "FNMR@1% FMR", "optimal threshold"]
        )
    return pd.DataFrame(rows)


def plot_path(name: str) -> str | None:
    path = APP_PLOTS_DIR / name
    return str(path) if path.exists() else None


STATIC_GRAPH_CACHE: dict[str, str] = {}


def static_graph_html(name: str, title: str) -> str:
    cache_key = f"{name}|{title}"
    if cache_key in STATIC_GRAPH_CACHE:
        return STATIC_GRAPH_CACHE[cache_key]

    path = APP_PLOTS_DIR / name
    if not path.exists():
        path = APP_PLOTS_DIR / "comparison" / "accuracy_4way_comparison.png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    html = f"""
    <div class="static-graph-card">
        <div class="static-graph-title">{title}</div>
        <img src="data:image/png;base64,{encoded}" alt="{title}" />
    </div>
    """
    STATIC_GRAPH_CACHE[cache_key] = html
    return html


def update_results_view(view_name: str):
    if view_name == BASELINE_RESULTS_VIEW:
        return (
            static_graph_html("accuracy_per_filter.png", "Accuracy per filter"),
            static_graph_html("det_curves.png", "DET curves"),
            static_graph_html("comparison/fnmr_improvement.png", "Mitigation comparison"),
        )
    if view_name == IMPROVED_RESULTS_VIEW:
        return (
            static_graph_html("comparison/accuracy_4way_comparison.png", "Accuracy per filter"),
            static_graph_html("comparison/fnmr_improvement.png", "DET / FNMR behavior"),
            static_graph_html("comparison/summary_table.png", "Mitigation comparison"),
        )
    return (
        static_graph_html("comparison/accuracy_4way_comparison.png", "Accuracy per filter"),
        static_graph_html("comparison/det_curves_comparison.png", "DET curves"),
        static_graph_html("comparison/summary_table.png", "Mitigation comparison"),
    )


def generate_report() -> str:
    cached_report = PROJECT_ROOT / "results" / "report" / "facial_filter_recognition_report.pdf"
    if cached_report.exists() and cached_report.stat().st_size > 0:
        return str(cached_report)
    report_module = load_report_module()
    return report_module.generate_report()


EMPTY_RESULT_HTML = "<div class='empty-state'>Verification result appears here</div>"
EMPTY_SIMILARITY_HTML = "<div class='empty-state compact'>Similarity breakdown appears here</div>"


def empty_similarity_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["condition", "similarity"])


def reset_verification_outputs():
    return [], EMPTY_RESULT_HTML, EMPTY_SIMILARITY_HTML, empty_similarity_df(), {}


def clear_image_one():
    return None, [], EMPTY_RESULT_HTML, EMPTY_SIMILARITY_HTML, empty_similarity_df(), {}


def clear_image_two():
    return None, [], EMPTY_RESULT_HTML, EMPTY_SIMILARITY_HTML, empty_similarity_df(), {}


def clear_all_images():
    return None, None, [], EMPTY_RESULT_HTML, EMPTY_SIMILARITY_HTML, empty_similarity_df(), {}


CUSTOM_CSS = """
:root {
    --bg:        #03070f;
    --surface:   #080e1c;
    --surface2:  #0c1428;
    --surface3:  #111d35;
    --line:      rgba(99, 161, 255, 0.10);
    --line2:     rgba(99, 161, 255, 0.18);
    --ink:       #e8f0ff;
    --ink2:      #8ba3cc;
    --ink3:      #4d6a99;
    --blue:      #3b82f6;
    --blue-glow: rgba(59, 130, 246, 0.35);
    --cyan:      #06b6d4;
    --cyan-glow: rgba(6, 182, 212, 0.25);
    --green:     #10b981;
    --green-glow:rgba(16, 185, 129, 0.25);
    --red:       #f43f5e;
    --amber:     #f59e0b;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --display: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* ─── reset & base ─── */
footer { display: none !important; }
*, *::before, *::after { box-sizing: border-box; }

.gradio-container {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: var(--display) !important;
    min-height: 100vh;
}

/* animated grid background */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background: linear-gradient(180deg, rgba(59,130,246,0.04), transparent 340px);
    pointer-events: none;
    z-index: 0;
}

.app-shell {
    max-width: 1540px;
    margin: 0 auto;
    padding: 28px 28px 60px;
    position: relative;
    z-index: 1;
}

/* ─── hero ─── */
.hero-band {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #050d1f 0%, #0a1529 50%, #060e20 100%);
    border: 1px solid var(--line2);
    border-radius: 16px;
    padding: 40px 44px;
    display: grid;
    grid-template-columns: 1.5fr 1fr;
    gap: 32px;
    align-items: center;
    box-shadow: 0 0 0 1px rgba(59,130,246,0.06), 0 32px 80px rgba(0,0,0,0.7);
    animation: heroIn 0.8s cubic-bezier(0.16,1,0.3,1) both;
}
@keyframes heroIn {
    from { opacity:0; transform: translateY(24px); }
    to   { opacity:1; transform: translateY(0); }
}

/* glowing orb behind hero */
.hero-band::before {
    content: '';
    position: absolute;
    width: 520px; height: 520px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    top: -180px; right: -80px;
    pointer-events: none;
}
.hero-band::after {
    content: '';
    position: absolute;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(6,182,212,0.10) 0%, transparent 70%);
    bottom: -100px; left: 20%;
    pointer-events: none;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #93c5fd;
    border: 1px solid rgba(59,130,246,0.35);
    background: rgba(59,130,246,0.08);
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: var(--mono);
    position: relative;
}
.eyebrow::before {
    content: '';
    width: 6px; height: 6px;
    background: var(--cyan);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--cyan);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.5; transform:scale(0.7); }
}

.hero-band h1 {
    font-size: 48px;
    line-height: 1;
    margin: 14px 0 12px;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ffffff 0%, #93c5fd 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-band > div:first-child > p {
    margin: 0;
    color: var(--ink2);
    font-size: 15px;
    line-height: 1.6;
    max-width: 540px;
    font-weight: 400;
}

.hero-meta {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}
.hero-meta div {
    border: 1px solid var(--line2);
    background: rgba(59,130,246,0.05);
    border-radius: 10px;
    padding: 14px 16px;
    transition: border-color 0.2s, background 0.2s;
}
.hero-meta div:hover {
    border-color: rgba(59,130,246,0.35);
    background: rgba(59,130,246,0.10);
}
.hero-meta span {
    color: var(--ink3);
    display: block;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: var(--mono);
}
.hero-meta strong {
    display: block;
    margin-top: 5px;
    font-size: 15px;
    color: var(--ink);
    font-weight: 600;
}

/* ─── KPI cards ─── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0,1fr));
    gap: 12px;
    margin: 20px 0;
}
.kpi-card {
    position: relative;
    overflow: hidden;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 20px 20px 16px;
    transition: transform 0.25s cubic-bezier(0.34,1.56,0.64,1), border-color 0.25s, box-shadow 0.25s;
    animation: cardIn 0.6s cubic-bezier(0.16,1,0.3,1) both;
}
.kpi-card:nth-child(1){animation-delay:0.05s}
.kpi-card:nth-child(2){animation-delay:0.10s}
.kpi-card:nth-child(3){animation-delay:0.15s}
.kpi-card:nth-child(4){animation-delay:0.20s}
.kpi-card:nth-child(5){animation-delay:0.25s}
.kpi-card:nth-child(6){animation-delay:0.30s}
.kpi-card:nth-child(7){animation-delay:0.35s}
.kpi-card:nth-child(8){animation-delay:0.40s}
@keyframes cardIn {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}
.kpi-card:hover {
    transform: translateY(-4px) scale(1.01);
    border-color: var(--line2);
    box-shadow: 0 0 30px var(--blue-glow), 0 20px 40px rgba(0,0,0,0.4);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.5), transparent);
    opacity: 0;
    transition: opacity 0.25s;
}
.kpi-card:hover::before { opacity: 1; }

.kpi-label {
    display: block;
    color: var(--ink3);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: var(--mono);
}
.kpi-card strong {
    display: block;
    margin: 10px 0 4px;
    font-size: 26px;
    line-height: 1;
    color: var(--ink);
    font-weight: 700;
    letter-spacing: -0.02em;
}
.kpi-card span:last-child {
    color: var(--ink3);
    font-size: 12px;
    font-family: var(--mono);
}

/* ─── section headings ─── */
.section-heading {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 16px;
    margin: 8px 0 16px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--line);
}
.section-heading h2 {
    margin: 0;
    font-size: 20px;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.01em;
}
.section-heading p {
    margin: 4px 0 0;
    color: var(--ink3);
    font-size: 13px;
    font-family: var(--mono);
}

/* ─── panels ─── */
.workspace-panel,
.control-panel,
.chart-panel,
.report-panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    transition: border-color 0.25s;
}
.workspace-panel:hover,
.control-panel:hover { border-color: var(--line2); }

.workspace-panel h3,
.control-panel h3,
.report-panel h3 {
    margin: 0 0 16px;
    font-size: 13px;
    font-weight: 600;
    color: var(--ink2);
    text-transform: uppercase;
    letter-spacing: 0.10em;
    font-family: var(--mono);
}

/* strip Gradio borders inside panels */
.control-panel .form,
.control-panel .wrap,
.workspace-panel .wrap {
    border: 0 !important;
    background: transparent !important;
}

/* ─── Gradio components dark overrides ─── */
.gradio-container input,
.gradio-container select,
.gradio-container textarea {
    background: var(--surface2) !important;
    border-color: var(--line2) !important;
    color: var(--ink) !important;
    font-family: var(--display) !important;
}
.gradio-container label {
    color: var(--ink2) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    font-family: var(--mono) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
/* radio buttons */
.gradio-container input[type=radio] { accent-color: var(--blue); }
.gradio-container .wrap.svelte-1jfk7uq,
.gradio-container .wrap { background: transparent !important; }

/* tabs: keep Gradio's native tab behavior, only polish the surface */
#main-tabs .tab-nav,
.gradio-container .tab-nav {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--line) !important;
    border-radius: 0 !important;
    position: relative !important;
    z-index: 50 !important;
    pointer-events: auto !important;
}
#main-tabs .tab-nav button,
.gradio-container .tab-nav button {
    pointer-events: auto !important;
    position: relative !important;
    z-index: 51 !important;
    color: var(--ink2) !important;
    font-family: var(--display) !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border-bottom: 3px solid transparent !important;
    padding: 14px 22px !important;
}
#main-tabs .tab-nav button[aria-selected="true"],
#main-tabs .tab-nav button.selected,
.gradio-container .tab-nav button[aria-selected="true"],
.gradio-container .tab-nav button.selected {
    color: #ffffff !important;
    background: rgba(59,130,246,0.12) !important;
    border-bottom-color: var(--blue) !important;
}
#main-tabs .tab-nav button:hover,
.gradio-container .tab-nav button:hover {
    color: #ffffff !important;
    background: rgba(59,130,246,0.10) !important;
}
#main-tabs .tabitem,
.gradio-container .tabitem {
    background: transparent !important;
    border: none !important;
    padding: 20px 0 0 !important;
}

/* radio choices: make selected model / mitigation unmistakable */
.control-panel input[type="radio"] {
    appearance: none !important;
    -webkit-appearance: none !important;
    width: 18px !important;
    height: 18px !important;
    min-width: 18px !important;
    border-radius: 50% !important;
    border: 2px solid #5f78a8 !important;
    background: #07142a !important;
    box-shadow: inset 0 0 0 4px #07142a !important;
    margin-right: 10px !important;
    vertical-align: middle !important;
}
.control-panel input[type="radio"]:checked {
    border-color: #60a5fa !important;
    background: #60a5fa !important;
    box-shadow: inset 0 0 0 4px #07142a, 0 0 0 4px rgba(59,130,246,0.25), 0 0 18px rgba(59,130,246,0.65) !important;
}
.control-panel label:has(input[type="radio"]) {
    border: 1px solid rgba(96,165,250,0.18) !important;
    border-radius: 10px !important;
    background: rgba(96,165,250,0.08) !important;
    color: var(--ink2) !important;
    transition: background 0.12s, border-color 0.12s, color 0.12s, box-shadow 0.12s !important;
}
.control-panel label:has(input[type="radio"]:checked) {
    border-color: rgba(96,165,250,0.75) !important;
    background: linear-gradient(135deg, rgba(37,99,235,0.34), rgba(6,182,212,0.18)) !important;
    color: #ffffff !important;
    box-shadow: 0 0 0 1px rgba(96,165,250,0.28), 0 0 22px rgba(59,130,246,0.25) !important;
}

/* image upload zones */
.gradio-container .upload-container,
.gradio-container .image-container {
    background: var(--surface2) !important;
    border: 1px dashed var(--line2) !important;
    border-radius: 10px !important;
}
.gradio-container .upload-container button,
.gradio-container .image-container button,
.gradio-container .image-container [role="button"],
.gradio-container .upload-container [role="button"] {
    pointer-events: auto !important;
    visibility: visible !important;
}

.clear-row {
    display: flex;
    gap: 10px;
    margin: 10px 0 14px;
}
.clear-row button {
    min-height: 34px !important;
    border-radius: 8px !important;
    background: var(--surface2) !important;
    border: 1px solid var(--line2) !important;
    color: var(--ink2) !important;
    font-size: 12px !important;
}

/* dropdown */
.gradio-container .dropdown-arrow { color: var(--blue) !important; }

/* ─── verify button ─── */
#verify-button,
#verify-button button {
    min-height: 52px !important;
    border-radius: 10px !important;
    font-family: var(--display) !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 50%, #06b6d4 100%) !important;
    border: none !important;
    box-shadow: 0 0 24px var(--blue-glow), 0 4px 16px rgba(0,0,0,0.4) !important;
    transition: box-shadow 0.25s, transform 0.2s !important;
    position: relative;
    overflow: hidden;
}
#verify-button button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.1), transparent);
    opacity: 0;
    transition: opacity 0.2s;
}
#verify-button button:hover::before { opacity: 1; }
#verify-button button:hover {
    box-shadow: 0 0 40px rgba(59,130,246,0.6), 0 8px 24px rgba(0,0,0,0.5) !important;
    transform: translateY(-1px) !important;
}
#verify-button button:active {
    transform: translateY(0) !important;
}

/* ─── result / decision card ─── */
.decision-card {
    border-radius: 12px;
    padding: 22px;
    border: 1px solid var(--line);
    background: var(--surface2);
    animation: slideUp 0.4s cubic-bezier(0.16,1,0.3,1) both;
}
@keyframes slideUp {
    from { opacity:0; transform:translateY(12px); }
    to   { opacity:1; transform:translateY(0); }
}
.decision-card.same {
    border-color: rgba(16,185,129,0.40);
    background: rgba(16,185,129,0.05);
    box-shadow: 0 0 32px rgba(16,185,129,0.12);
}
.decision-card.different {
    border-color: rgba(244,63,94,0.40);
    background: rgba(244,63,94,0.05);
    box-shadow: 0 0 32px rgba(244,63,94,0.12);
}

.decision-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 4px;
}
.decision-header span {
    color: var(--ink3);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: var(--mono);
}
.decision-header strong {
    font-size: 18px;
    font-weight: 700;
    color: var(--ink);
}

.decision-score {
    font-size: 64px;
    line-height: 1;
    font-weight: 800;
    letter-spacing: -0.04em;
    margin: 16px 0;
    background: linear-gradient(135deg, #ffffff, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: var(--mono);
}

.decision-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 14px;
}
.decision-grid div {
    background: var(--surface3);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 10px 12px;
}
.decision-grid span {
    color: var(--ink3);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    font-family: var(--mono);
}
.decision-grid strong {
    display: block;
    margin-top: 4px;
    font-size: 15px;
    color: var(--ink);
    font-weight: 600;
}
.decision-card p {
    margin: 12px 0 0;
    color: var(--ink3);
    font-size: 12px;
    font-family: var(--mono);
}

/* ─── score / similarity tracks ─── */
.score-track,
.similarity-track,
.risk-meter {
    height: 6px;
    border-radius: 100px;
    background: var(--surface3);
    overflow: hidden;
}
.score-track { margin-top: 14px; }
.score-track span,
.similarity-track span {
    display: block;
    height: 100%;
    border-radius: 100px;
    transition: width 0.6s cubic-bezier(0.34,1.56,0.64,1);
    box-shadow: 0 0 8px currentColor;
}
.risk-meter span {
    display: block;
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--green), var(--amber), var(--red));
    transition: width 0.6s cubic-bezier(0.34,1.56,0.64,1);
}

/* ─── similarity / risk panels ─── */
.similarity-panel,
.risk-panel {
    background: var(--surface2);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 16px;
}
.similarity-row,
.risk-row {
    display: grid;
    grid-template-columns: minmax(160px,.9fr) 1.2fr;
    gap: 16px;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid var(--line);
}
.similarity-row:last-child,
.risk-row:last-child { border-bottom: 0; }
.similarity-row div:first-child,
.risk-row div:first-child {
    display: flex;
    flex-direction: column;
    gap: 3px;
}
.similarity-row strong,
.risk-row strong {
    font-size: 13px;
    font-weight: 600;
    color: var(--ink);
}
.similarity-row span,
.risk-row span {
    color: var(--ink3);
    font-size: 11px;
    font-family: var(--mono);
}

/* ─── empty states ─── */
.empty-state {
    border: 1px dashed var(--line2);
    background: var(--surface2);
    color: var(--ink3);
    border-radius: 10px;
    padding: 28px 20px;
    text-align: center;
    font-size: 13px;
    font-family: var(--mono);
}
.empty-state.compact { padding: 16px; }

/* ─── chart / report images ─── */
.chart-panel img,
.report-panel img {
    border-radius: 8px;
    border: 1px solid var(--line);
}
.static-graph-card {
    width: 100%;
}
.static-graph-title {
    margin: 0 0 12px;
    font-size: 13px;
    font-weight: 600;
    color: var(--ink2);
    text-transform: uppercase;
    letter-spacing: 0.10em;
    font-family: var(--mono);
}
.static-graph-card img {
    display: block;
    width: 100%;
    max-height: 520px;
    object-fit: contain;
    background: #ffffff;
}

/* ─── dataframe ─── */
.dataframe {
    border-radius: 10px !important;
    background: var(--surface2) !important;
    border: 1px solid var(--line) !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
}
.dataframe thead { background: var(--surface3) !important; }
.dataframe th {
    color: var(--ink3) !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 10px 14px !important;
}
.dataframe td {
    color: var(--ink2) !important;
    padding: 8px 14px !important;
    border-bottom: 1px solid var(--line) !important;
}
.dataframe tr:hover td { background: var(--surface3) !important; }

/* secondary report button */
.gradio-container button.secondary {
    background: var(--surface2) !important;
    border: 1px solid var(--line2) !important;
    color: var(--ink2) !important;
    font-family: var(--display) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    transition: border-color 0.2s, color 0.2s, box-shadow 0.2s !important;
}
.gradio-container button.secondary:hover {
    border-color: var(--blue) !important;
    color: var(--ink) !important;
    box-shadow: 0 0 16px var(--blue-glow) !important;
}

/* gallery */
.gradio-container .gallery {
    background: var(--surface2) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--line2); }

/* ─── responsive ─── */
@media (max-width: 980px) {
    .hero-band { grid-template-columns: 1fr; }
    .kpi-grid  { grid-template-columns: 1fr 1fr; }
    .hero-band h1 { font-size: 36px; }
}

/* performance overrides for smoother tab and mode switching */
.hero-band,
.kpi-card,
.decision-card {
    animation: none !important;
}
.hero-band::before,
.hero-band::after {
    display: none !important;
}
.gradio-container *,
.gradio-container *::before,
.gradio-container *::after {
    transition-duration: 0.12s !important;
    animation-duration: 0.12s !important;
    animation-iteration-count: 1 !important;
}
"""


with gr.Blocks(
    title="FaceShield Analytics",
) as demo:
    with gr.Column(elem_classes="app-shell"):
        gr.HTML(
            f"""
            <section class="hero-band">
                <div>
                    <span class="eyebrow">Facial Filter Recognition Lab &mdash; IJCB 2025</span>
                    <h1>FaceShield Analytics</h1>
                    <p>End-to-end face recognition stress testing under AR filters &mdash; with baseline and improved-framework comparison. Based on Ozturk et al., 2025.</p>
                </div>
                <div class="hero-meta">
                    <div><span>Dataset</span><strong>LFW Benchmark</strong></div>
                    <div><span>Track</span><strong>Improved / Baseline</strong></div>
                    <div><span>Mitigation</span><strong>Ridge Regression</strong></div>
                    <div><span>Filter Classes</span><strong>6 AR Types</strong></div>
                </div>
            </section>
            """
        )
        gr.HTML(dashboard_cards_html())

        gr.HTML(
            """
            <div class="section-heading">
                <div>
                    <h2>Live Verification Console</h2>
                    <p>Compare two faces under filter and mitigation settings.</p>
                </div>
            </div>
            """
        )
        with gr.Row():
            with gr.Column(scale=7, elem_classes="workspace-panel"):
                gr.HTML("<h3>Image Intake</h3>")
                with gr.Row():
                    img1 = gr.Image(
                        label="Image 1",
                        type="pil",
                        height=290,
                        image_mode="RGB",
                        sources=["upload", "webcam", "clipboard"],
                        interactive=True,
                    )
                    img2 = gr.Image(
                        label="Image 2",
                        type="pil",
                        height=290,
                        image_mode="RGB",
                        sources=["upload", "webcam", "clipboard"],
                        interactive=True,
                    )
                with gr.Row(elem_classes="clear-row"):
                    clear_img1_btn = gr.Button("Clear Image 1", variant="secondary", size="sm")
                    clear_img2_btn = gr.Button("Clear Image 2", variant="secondary", size="sm")
                    clear_all_btn = gr.Button("Clear Both", variant="secondary", size="sm")
                gallery = gr.Gallery(label="Filtered images", columns=2, height=340)

            with gr.Column(scale=5, elem_classes="control-panel"):
                gr.HTML("<h3>Recognition Controls</h3>")
                model_radio = gr.Radio(
                    choices=[BASELINE_MODEL_CHOICE, IMPROVED_MODEL_CHOICE],
                    value=IMPROVED_MODEL_CHOICE,
                    label="Recognition model",
                    interactive=True,
                )
                mitigation_radio = gr.Radio(
                    choices=["None", PAPER_MITIGATION_CHOICE, IMPROVED_MITIGATION_CHOICE],
                    value=IMPROVED_MITIGATION_CHOICE,
                    label="Mitigation",
                    interactive=True,
                )
                filter_dropdown = gr.Dropdown(
                    choices=["none"] + FILTER_NAMES,
                    value="none",
                    label="Filter",
                )
                verify_button = gr.Button("Verify Identity", variant="primary", elem_id="verify-button")
                result_panel = gr.HTML(EMPTY_RESULT_HTML)
                similarity_panel = gr.HTML(EMPTY_SIMILARITY_HTML)

        comparison_df = gr.DataFrame(value=empty_similarity_df, label="Similarity ledger", interactive=False)
        embedding_state = gr.State({})
        verify_button.click(
            verify_identity,
            inputs=[img1, img2, model_radio, mitigation_radio, filter_dropdown, embedding_state],
            outputs=[gallery, result_panel, similarity_panel, comparison_df, embedding_state],
            trigger_mode="always_last",
            concurrency_limit=1,
        )
        for control in [model_radio, mitigation_radio, filter_dropdown, img1, img2]:
            control.change(
                reset_verification_outputs,
                outputs=[gallery, result_panel, similarity_panel, comparison_df, embedding_state],
                queue=False,
                show_progress="hidden",
            )
        clear_img1_btn.click(
            clear_image_one,
            outputs=[img1, gallery, result_panel, similarity_panel, comparison_df, embedding_state],
            queue=False,
            show_progress="hidden",
        )
        clear_img2_btn.click(
            clear_image_two,
            outputs=[img2, gallery, result_panel, similarity_panel, comparison_df, embedding_state],
            queue=False,
            show_progress="hidden",
        )
        clear_all_btn.click(
            clear_all_images,
            outputs=[img1, img2, gallery, result_panel, similarity_panel, comparison_df, embedding_state],
            queue=False,
            show_progress="hidden",
        )

        gr.HTML(
            """
            <div class="section-heading">
                <div>
                    <h2>Evaluation Analytics</h2>
                    <p>Model comparison, DET behavior, mitigation lift, and filter risk profile.</p>
                </div>
            </div>
            """
        )
        with gr.Row():
            results_view = gr.Dropdown(
                choices=[
                    BASELINE_RESULTS_VIEW,
                    IMPROVED_RESULTS_VIEW,
                    "Side-by-side comparison",
                ],
                value="Side-by-side comparison",
                label="View results for",
            )
        with gr.Row():
            with gr.Column(elem_classes="chart-panel"):
                accuracy_plot = gr.HTML(
                    value=static_graph_html("comparison/accuracy_4way_comparison.png", "Accuracy per filter")
                )
            with gr.Column(elem_classes="chart-panel"):
                det_plot = gr.HTML(value=static_graph_html("comparison/det_curves_comparison.png", "DET curves"))
        with gr.Row():
            with gr.Column(scale=7, elem_classes="chart-panel"):
                mitigation_plot = gr.HTML(
                    value=static_graph_html("comparison/summary_table.png", "Mitigation comparison")
                )
            with gr.Column(scale=5, elem_classes="chart-panel"):
                gr.HTML("<h3>Filter Risk Profile</h3>")
                gr.HTML(filter_impact_html())
        gr.DataFrame(value=metrics_table, label="Metrics summary", interactive=False)
        results_view.change(
            update_results_view,
            inputs=[results_view],
            outputs=[accuracy_plot, det_plot, mitigation_plot],
        )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Base(
            font=[gr.themes.Font("Inter"), gr.themes.Font("sans-serif")],
            font_mono=[gr.themes.Font("SFMono-Regular"), gr.themes.Font("monospace")],
        ),
        css=CUSTOM_CSS,
        show_error=True,
    )
