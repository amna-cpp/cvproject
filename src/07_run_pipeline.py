from __future__ import annotations

import argparse
import importlib.util
import time
from datetime import datetime
from pathlib import Path

from common import FILTER_NAMES, METRICS_DIR, PROJECT_ROOT, parse_models, read_json


SUBSET_SIZE = 300

CONFIG = {
    "project_root": PROJECT_ROOT,
    "subset_size": SUBSET_SIZE,
    "models": ["arcface", "facenet"],
}


def load_step(filename: str):
    path = Path(__file__).resolve().parent / filename
    module_name = f"pipeline_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log_step(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def print_summary() -> None:
    recognition = read_json(METRICS_DIR / "recognition_metrics.json", default={})
    mitigation = read_json(METRICS_DIR / "mitigation_metrics.json", default={}).get("filters", {})

    print("\nFinal summary")
    print("Filter | Accuracy | FNMR@1% | Corrected Accuracy")
    original = recognition.get("original", {})
    print(
        f"original | {original.get('accuracy', float('nan')):.2%} | "
        f"{original.get('fnmr_fmr01', float('nan')):.2%} | N/A"
    )
    for filter_name in FILTER_NAMES:
        metrics = recognition.get(filter_name, {})
        corrected = mitigation.get(filter_name, {})
        print(
            f"{filter_name} | {metrics.get('accuracy', float('nan')):.2%} | "
            f"{metrics.get('fnmr_fmr01', float('nan')):.2%} | "
            f"{corrected.get('corrected_accuracy', float('nan')):.2%}"
        )


def run(subset_size: int = SUBSET_SIZE, models: list[str] | str = "arcface,facenet") -> None:
    start_time = time.time()
    model_keys = parse_models(models)
    CONFIG["subset_size"] = subset_size
    CONFIG["models"] = model_keys

    setup = load_step("01_setup_dataset.py")
    filters = load_step("02_apply_filters.py")
    selection = load_step("03_filter_selection.py")
    embeddings = load_step("04_extract_embeddings.py")
    evaluate = load_step("05_evaluate_recognition.py")
    mitigation = load_step("06_mitigation.py")

    log_step("Step 1: dataset setup")
    setup.run(subset_size=subset_size)
    log_step("Step 1 complete")

    log_step("Step 2: filter application")
    filters.run(subset_size=subset_size)
    log_step("Step 2 complete")

    log_step("Step 3: filter impact ranking")
    selection.run(sample_size=50)
    log_step("Step 3 complete")

    log_step(f"Step 4: embedding extraction for {', '.join(model_keys)}")
    embeddings.run(models=model_keys, subset_size=subset_size)
    log_step("Step 4 complete")

    log_step("Step 5: recognition evaluation")
    for model_key in model_keys:
        evaluate.run(model_name=model_key)
    log_step("Step 5 complete")

    if "arcface" in model_keys:
        log_step("Step 6: mitigation")
        mitigation.run()
        log_step("Step 6 complete")
    else:
        log_step("Step 6 skipped: mitigation is implemented for ArcFace embeddings")

    print_summary()
    print(f"Total elapsed time: {time.time() - start_time:.1f}s")


def run_upgrade_only(subset_size: int = SUBSET_SIZE) -> None:
    start_time = time.time()
    CONFIG["subset_size"] = subset_size
    adaface_embeddings = load_step("04b_extract_embeddings_adaface.py")
    evaluate = load_step("05_evaluate_recognition.py")
    linear_mitigation = load_step("06b_mitigation_linear.py")
    compare = load_step("08_compare_models.py")
    report = load_step("09_generate_report.py")

    log_step("Step 8: AdaFace/GhostFaceNet embedding extraction")
    adaface_embeddings.run(subset_size=subset_size)
    log_step("Step 8 complete")

    log_step("Step 9: AdaFace/GhostFaceNet recognition evaluation")
    evaluate.run(model_name="adaface")
    log_step("Step 9 complete")

    log_step("Step 10: linear mitigation for ArcFace and AdaFace")
    linear_mitigation.run(backbone="both")
    log_step("Step 10 complete")

    log_step("Step 11: model comparison plots")
    compare.run()
    log_step("Step 11 complete")

    log_step("Step 12: PDF report generation")
    report.generate_report()
    log_step("Step 12 complete")
    print(f"Total elapsed time: {time.time() - start_time:.1f}s")


def run_report_only() -> None:
    start_time = time.time()
    report = load_step("09_generate_report.py")
    log_step("Report-only generation")
    report.generate_report()
    log_step("Report-only complete")
    print(f"Total elapsed time: {time.time() - start_time:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full facial filter impact pipeline.")
    parser.add_argument("--subset", type=int, default=SUBSET_SIZE, help="Pairs per class to evaluate.")
    parser.add_argument("--models", default="arcface,facenet", help="Comma-separated models: arcface,facenet")
    parser.add_argument("--upgrade-only", action="store_true", help="Skip steps 1-7 and run only upgrade steps 8-12.")
    parser.add_argument("--report-only", action="store_true", help="Skip all computation and regenerate the PDF report.")
    args = parser.parse_args()
    if args.report_only:
        run_report_only()
    elif args.upgrade_only:
        run_upgrade_only(subset_size=args.subset)
    else:
        run(subset_size=args.subset, models=args.models)


if __name__ == "__main__":
    main()
