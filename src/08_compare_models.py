from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COMPARISON_COLORS, FILTER_NAMES, METRICS_DIR, PROJECT_ROOT, read_json


CONFIG = {
    "ARCFACE_METRICS": "results/metrics/recognition_metrics.json",
    "ADAFACE_METRICS": "results/metrics/recognition_metrics_adaface.json",
    "MITIG_OLD": "results/metrics/mitigation_metrics.json",
    "MITIG_NEW": "results/mitigation_linear/linear_mitigation_metrics.json",
    "OUTPUT_DIR": "results/plots/comparison/",
}


def project_path(value: str) -> Path:
    return PROJECT_ROOT / value


def safe_metric(payload: dict, path: list[str], default: float = float("nan")) -> float:
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def load_payloads() -> tuple[dict, dict, dict, dict]:
    return (
        read_json(project_path(CONFIG["ARCFACE_METRICS"]), default={}),
        read_json(project_path(CONFIG["ADAFACE_METRICS"]), default={}),
        read_json(project_path(CONFIG["MITIG_OLD"]), default={}),
        read_json(project_path(CONFIG["MITIG_NEW"]), default={}),
    )


def comparison_values() -> pd.DataFrame:
    arcface, adaface, old_mitig, new_mitig = load_payloads()
    old_filters = old_mitig.get("filters", {})
    rows = []
    for filter_name in FILTER_NAMES:
        rows.append(
            {
                "Filter": filter_name,
                "ArcFace no correction": safe_metric(arcface, [filter_name, "accuracy"]),
                "ArcFace mean-shift": safe_metric(
                    old_filters,
                    [filter_name, "corrected_accuracy"],
                    safe_metric(new_mitig, ["arcface", filter_name, "accuracy_meanshift"]),
                ),
                "AdaFace no correction": safe_metric(
                    adaface,
                    [filter_name, "accuracy"],
                    safe_metric(new_mitig, ["adaface", filter_name, "accuracy_filtered"]),
                ),
                "AdaFace linear": safe_metric(new_mitig, ["adaface", filter_name, "accuracy_linear"]),
                "Paper FNMR": safe_metric(arcface, [filter_name, "fnmr_fmr01"]),
                "+MeanShift": safe_metric(
                    old_filters,
                    [filter_name, "corrected_fnmr_fmr01"],
                    safe_metric(new_mitig, ["arcface", filter_name, "fnmr_fmr1_meanshift"]),
                ),
                "AdaFace": safe_metric(
                    adaface,
                    [filter_name, "fnmr_fmr01"],
                    safe_metric(new_mitig, ["adaface", filter_name, "fnmr_fmr1_filtered"]),
                ),
                "+LinearCorr": safe_metric(new_mitig, ["adaface", filter_name, "fnmr_fmr1_linear"]),
                "Improvement%": safe_metric(new_mitig, ["adaface", filter_name, "improvement_pct"]),
            }
        )
    return pd.DataFrame(rows)


def value_label(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        if np.isfinite(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )


def plot_accuracy(df: pd.DataFrame) -> None:
    output_dir = project_path(CONFIG["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = ["ArcFace no correction", "ArcFace mean-shift", "AdaFace no correction", "AdaFace linear"]
    colors = [
        COMPARISON_COLORS["arcface_none"],
        COMPARISON_COLORS["arcface_meanshift"],
        COMPARISON_COLORS["adaface_none"],
        COMPARISON_COLORS["adaface_linear"],
    ]
    x = np.arange(len(df))
    width = 0.2
    _, ax = plt.subplots(figsize=(12, 5))
    for idx, (method, color) in enumerate(zip(methods, colors)):
        bars = ax.bar(x + (idx - 1.5) * width, df[method].to_numpy(dtype=float), width, label=method, color=color)
        value_label(ax, bars)
    arcface, _, _, _ = load_payloads()
    baseline = safe_metric(arcface, ["original", "accuracy"])
    if np.isfinite(baseline):
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1, label="Original accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Filter"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy under facial filters: paper vs improved framework")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_4way_comparison.png", dpi=200)
    plt.close()


def plot_fnmr(df: pd.DataFrame) -> None:
    output_dir = project_path(CONFIG["OUTPUT_DIR"])
    methods = ["Paper FNMR", "+MeanShift", "AdaFace", "+LinearCorr"]
    colors = [
        COMPARISON_COLORS["arcface_none"],
        COMPARISON_COLORS["arcface_meanshift"],
        COMPARISON_COLORS["adaface_none"],
        COMPARISON_COLORS["adaface_linear"],
    ]
    plt.figure(figsize=(10, 5))
    for method, color in zip(methods, colors):
        plt.plot(df["Filter"], df[method].to_numpy(dtype=float), marker="o", label=method, color=color)
    plt.ylabel("FNMR@FMR=1% (lower is better)")
    plt.xlabel("Filter")
    plt.title("FNMR@FMR=1%: paper vs improved framework")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fnmr_improvement.png", dpi=200)
    plt.close()


def most_disruptive_filter(df: pd.DataFrame) -> str:
    values = df["Paper FNMR"].to_numpy(dtype=float)
    if np.all(~np.isfinite(values)):
        return FILTER_NAMES[0]
    return str(df.iloc[int(np.nanargmax(values))]["Filter"])


def load_det_curve(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if path.exists():
        det = pd.read_csv(path)
        if {"fmr", "fnmr"}.issubset(det.columns):
            return np.clip(det["fmr"].to_numpy(dtype=float), 1e-4, 1.0), np.clip(det["fnmr"].to_numpy(dtype=float), 1e-4, 1.0)
    return None


def plot_det(df: pd.DataFrame) -> None:
    output_dir = project_path(CONFIG["OUTPUT_DIR"])
    filter_name = most_disruptive_filter(df)
    row = df[df["Filter"] == filter_name].iloc[0]
    curves = [
        ("ArcFace no correction", METRICS_DIR / f"det_curve_arcface_{filter_name}.csv", row["Paper FNMR"], COMPARISON_COLORS["arcface_none"]),
        ("ArcFace mean-shift", METRICS_DIR / f"det_curve_arcface_{filter_name}_corrected.csv", row["+MeanShift"], COMPARISON_COLORS["arcface_meanshift"]),
        ("AdaFace no correction", METRICS_DIR / f"det_curve_adaface_{filter_name}.csv", row["AdaFace"], COMPARISON_COLORS["adaface_none"]),
        ("AdaFace linear", METRICS_DIR / f"det_curve_adaface_{filter_name}_linear.csv", row["+LinearCorr"], COMPARISON_COLORS["adaface_linear"]),
    ]
    plt.figure(figsize=(8, 6))
    for label, path, _fallback_y, color in curves:
        loaded = load_det_curve(path)
        if loaded is None:
            continue
        fmr, fnmr = loaded
        plt.plot(fmr, fnmr, label=label, color=color, linewidth=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(0.001, 1.0)
    plt.ylim(0.001, 1.0)
    plt.xlabel("False Match Rate (FMR)")
    plt.ylabel("False Non-Match Rate (FNMR)")
    plt.title(f"DET curves — {filter_name}")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "det_curves_comparison.png", dpi=200)
    plt.close()


def plot_summary_table(df: pd.DataFrame) -> None:
    output_dir = project_path(CONFIG["OUTPUT_DIR"])
    table_df = df[["Filter", "Paper FNMR", "+MeanShift", "AdaFace", "+LinearCorr", "Improvement%"]].copy()
    table_df.to_csv(output_dir / "summary_table.csv", index=False, encoding="utf-8")
    display_df = table_df.copy()
    for col in display_df.columns[1:]:
        display_df[col] = display_df[col].map(lambda value: "" if not np.isfinite(value) else f"{value:.2%}" if col != "Improvement%" else f"{value:.1f}%")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#B5D4F4")
        if row == 0:
            cell.set_facecolor("#E6F1FB")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F8F8F8")
        if row > 0:
            source = table_df.iloc[row - 1]
            if col == 5 and np.isfinite(source["Improvement%"]) and source["Improvement%"] > 10:
                cell.set_facecolor("#EAF3DE")
            if col == 3 and np.isfinite(source["AdaFace"]) and np.isfinite(source["Paper FNMR"]) and source["AdaFace"] > source["Paper FNMR"]:
                cell.set_facecolor("#FCEBEB")
    ax.set_title("Filter | Paper FNMR | +MeanShift | AdaFace | +LinearCorr | Improvement%", pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / "summary_table.png", dpi=200)
    plt.close()


def pct_reduction(before: np.ndarray, after: np.ndarray) -> float:
    mask = np.isfinite(before) & np.isfinite(after) & (before > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.mean((before[mask] - after[mask]) / before[mask] * 100.0))


def print_summary(df: pd.DataFrame) -> None:
    paper = df["Paper FNMR"].to_numpy(dtype=float)
    meanshift = df["+MeanShift"].to_numpy(dtype=float)
    adaface = df["AdaFace"].to_numpy(dtype=float)
    linear = df["+LinearCorr"].to_numpy(dtype=float)
    combined = (paper - linear) / np.clip(paper, 1e-12, None) * 100.0
    finite_combined = np.where(np.isfinite(combined), combined, np.nan)
    best_idx = int(np.nanargmax(finite_combined)) if np.any(np.isfinite(finite_combined)) else 0
    worst_idx = int(np.nanargmin(finite_combined)) if np.any(np.isfinite(finite_combined)) else 0
    underperforms = int(np.sum((adaface > paper) & np.isfinite(adaface) & np.isfinite(paper)))
    print("══════════════════════════════════════════")
    print("IMPROVEMENT SUMMARY")
    print("══════════════════════════════════════════")
    print(f"Avg FNMR reduction — AdaFace vs ArcFace:          {pct_reduction(paper, adaface):.1f}%")
    print(f"Avg FNMR reduction — Linear vs Mean-shift:         {pct_reduction(meanshift, linear):.1f}%")
    print(f"Combined improvement (AdaFace+Linear vs Paper):    {pct_reduction(paper, linear):.1f}%")
    print(f"Best filter improved:  {df.iloc[best_idx]['Filter']}  ({finite_combined[best_idx]:.1f}% FNMR reduction)")
    print(f"Worst filter improved: {df.iloc[worst_idx]['Filter']}  ({finite_combined[worst_idx]:.1f}% FNMR reduction)")
    print(f"Cases where AdaFace underperforms ArcFace: {underperforms}/6 filters")
    print("══════════════════════════════════════════")


def run() -> pd.DataFrame:
    output_dir = project_path(CONFIG["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    df = comparison_values()
    plot_accuracy(df)
    plot_fnmr(df)
    plot_det(df)
    plot_summary_table(df)
    print_summary(df)
    return df


if __name__ == "__main__":
    run()
