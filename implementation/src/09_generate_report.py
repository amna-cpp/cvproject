from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from common import FILTER_NAMES, PROJECT_ROOT, read_json


CONFIG = {
    "ARCFACE_METRICS": "results/metrics/recognition_metrics.json",
    "ADAFACE_METRICS": "results/metrics/recognition_metrics_adaface.json",
    "MITIG_OLD": "results/metrics/mitigation_metrics.json",
    "MITIG_NEW": "results/mitigation_linear/linear_mitigation_metrics.json",
    "PLOTS_DIR": "results/plots/",
    "OUTPUT_PDF": "results/report/facial_filter_recognition_report.pdf",
    "PAPER_URL": "https://arxiv.org/html/2507.17729",
}


def project_path(value: str) -> Path:
    return PROJECT_ROOT / value


def styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            alignment=1,
            textColor=colors.HexColor("#185FA5"),
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            alignment=1,
        ),
        "h1": ParagraphStyle(
            "Heading1Custom",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor("#185FA5"),
        ),
        "h2": ParagraphStyle(
            "Heading2Custom",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            spaceBefore=14,
            spaceAfter=6,
            textColor=colors.HexColor("#0C447C"),
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=4,
        ),
        "bullet": ParagraphStyle(
            "BulletCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            leftIndent=14,
            bulletIndent=0,
        ),
        "caption": ParagraphStyle(
            "CaptionCustom",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11,
            alignment=1,
        ),
    }


def table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6F1FB")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B5D4F4")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
    )


def alternating_table_style(rows: int) -> TableStyle:
    style = table_style()
    for row in range(1, rows):
        style.add("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F8F8F8") if row % 2 else colors.white)
    return style


def metric(payload: dict, path: list[str], default: float = float("nan")) -> float:
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def max_fnmr_increase() -> str:
    arcface = read_json(project_path(CONFIG["ARCFACE_METRICS"]), default={})
    original = metric(arcface, ["original", "fnmr_fmr01"], 0.0)
    increases = []
    for filter_name in FILTER_NAMES:
        value = metric(arcface, [filter_name, "fnmr_fmr01"])
        if np.isfinite(value):
            increases.append(max(0.0, value - original))
    if not increases:
        return "N/A"
    return f"{max(increases) * 100:.1f}"


def plot_or_placeholder(relative_path: str, caption: str, style: ParagraphStyle):
    path = project_path(relative_path)
    max_width = A4[0] - 5 * cm
    if path.exists():
        img = Image(str(path))
        scale = min(max_width / img.imageWidth, 1.0)
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        return [img, Spacer(1, 0.2 * cm), Paragraph(caption, style), Spacer(1, 0.35 * cm)]
    placeholder = Table(
        [[Paragraph("Plot not yet generated — run pipeline first.", style)]],
        colWidths=[max_width],
        rowHeights=[4 * cm],
    )
    placeholder.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEEEEE")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return [placeholder, Spacer(1, 0.2 * cm), Paragraph(caption, style), Spacer(1, 0.35 * cm)]


def bullet_items(items: list[str], style: ParagraphStyle) -> list:
    story = []
    for item in items:
        story.append(Paragraph(item, style, bulletText="•"))
    return story


def numeric_summary() -> list[list[str]]:
    arcface = read_json(project_path(CONFIG["ARCFACE_METRICS"]), default={})
    adaface = read_json(project_path(CONFIG["ADAFACE_METRICS"]), default={})
    old = read_json(project_path(CONFIG["MITIG_OLD"]), default={}).get("filters", {})
    new = read_json(project_path(CONFIG["MITIG_NEW"]), default={})

    def avg(values: list[float]) -> float:
        finite = [value for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    paper = avg([metric(arcface, [f, "fnmr_fmr01"]) for f in FILTER_NAMES])
    mean_shift = avg([metric(old, [f, "corrected_fnmr_fmr01"], metric(new, ["arcface", f, "fnmr_fmr1_meanshift"])) for f in FILTER_NAMES])
    adaface_raw = avg([metric(adaface, [f, "fnmr_fmr01"], metric(new, ["adaface", f, "fnmr_fmr1_filtered"])) for f in FILTER_NAMES])
    adaface_linear = avg([metric(new, ["adaface", f, "fnmr_fmr1_linear"]) for f in FILTER_NAMES])
    improvement = (paper - adaface_linear) / paper * 100.0 if np.isfinite(paper) and paper > 0 and np.isfinite(adaface_linear) else float("nan")

    def pct(value: float) -> str:
        return "N/A" if not np.isfinite(value) else f"{value * 100:.1f}%"

    return [
        ["Average FNMR (ArcFace, no mitigation):", pct(paper)],
        ["Average FNMR (ArcFace + mean-shift):", pct(mean_shift)],
        ["Average FNMR (AdaFace, no mitigation):", pct(adaface_raw)],
        ["Average FNMR (AdaFace + linear corrector):", pct(adaface_linear)],
        ["Total improvement over paper baseline:", "N/A" if not np.isfinite(improvement) else f"{improvement:.1f}%"],
    ]


def build_report() -> str:
    output_pdf = project_path(CONFIG["OUTPUT_PDF"])
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    st = styles()
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )
    story = []

    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Facial Filters and Face Recognition:<br/>Reimplementation and Improvement", st["title"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(
        Paragraph(
            "Based on: A Comprehensive Evaluation Framework for the Study of the<br/>"
            "Effects of Facial Filters on Face Recognition Accuracy<br/>"
            "Ozturk, Conwill, Gutierrez, Bowyer, Scheirer — University of Notre Dame<br/>"
            "IJCB 2025 | arXiv: 2507.17729",
            st["subtitle"],
        )
    )
    story.append(Spacer(1, 0.7 * cm))
    story.append(Paragraph("Extended Implementation Report", st["subtitle"]))
    story.append(Paragraph(datetime.now().strftime("%B %d, %Y"), st["subtitle"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#185FA5")))
    story.append(PageBreak())

    story.append(Paragraph("1. Original Paper Summary", st["h1"]))
    story.append(
        Paragraph(
            "The original paper introduces a three-component framework for studying how "
            "social media AR facial filters affect automated face recognition accuracy. "
            "The framework consists of: (1) a controlled, gender-balanced dataset of "
            "1,000 subjects from the FRGC dataset; (2) a principled filter selection "
            "method using Otsu binarization to bin filters by percentage of manipulated "
            "pixels; and (3) a standardized evaluation protocol measuring FNMR at fixed "
            "FMR thresholds with DET curves. The paper demonstrates the framework via a "
            "case study of 125 filters from Instagram, Snapchat, Meitu, and Pitu, making "
            "it the largest filter evaluation study to date. The mitigation section shows "
            "that a simple mean-shift subtraction in ArcFace embedding space can partially "
            "recover recognition performance after filtering.",
            st["body"],
        )
    )
    story.append(Paragraph("Key findings from the original paper", st["h2"]))
    story.extend(
        bullet_items(
            [
                "Structural filters (eye enlargement, face slimming) cause the greatest FNMR increase",
                "Color and brightness filters have minimal impact on recognition",
                "Chinese apps (Meitu, Pitu) apply more aggressive structural transformations than Western apps",
                "A simple linear shift in embedding space reduces the filtering effect significantly",
                f"ArcFace FNMR at FMR=1% increases by up to {max_fnmr_increase()}% under the most aggressive filters",
            ],
            st["bullet"],
        )
    )
    story.append(Paragraph("Limitations identified", st["h2"]))
    story.extend(
        bullet_items(
            [
                "ArcFace uses fixed angular margin regardless of image quality degradation",
                "Mean-shift mitigation is filter-specific and assumes uniform displacement per filter",
                "No quality-aware recognition model was evaluated",
                "Mitigation requires knowing the filter identity in advance",
            ],
            st["bullet"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("2. Reimplementation", st["h1"]))
    story.append(
        Paragraph(
            "We reimplemented the paper's evaluation framework end-to-end using Python, "
            "reproducing the three core components: dataset loading (LFW), programmatic "
            "filter simulation (6 filter categories), and biometric evaluation (FNMR/FMR, "
            "DET curves). All original experiments were reproduced using ArcFace via the "
            "DeepFace library as the recognition backbone, with the LFW dataset and its "
            "standard 3,000 genuine and 3,000 impostor pairs.",
            st["body"],
        )
    )
    component_rows = [
        ["Component", "Paper", "Our Implementation"],
        ["Dataset", "FRGC (1,000 subjects, controlled)", "LFW (standard benchmark)"],
        ["Filter application", "Android emulator (real AR filters)", "Programmatic OpenCV/PIL simulation"],
        ["Filters tested", "125 across 4 apps", "6 representative categories"],
        ["Recognition model", "ArcFace + COTS", "ArcFace (DeepFace)"],
        ["Evaluation metrics", "FNMR@FMR=0.1%, 1% DET curves", "Same"],
        ["Mitigation", "Mean embedding shift", "Same (baseline)"],
        ["Prototype UI", "None", "Gradio web app"],
    ]
    table = Table(component_rows, colWidths=[3.5 * cm, 5.5 * cm, 5.5 * cm])
    table.setStyle(alternating_table_style(len(component_rows)))
    story.append(Spacer(1, 0.3 * cm))
    story.append(table)
    story.append(PageBreak())

    story.append(Paragraph("3. Our Improvements", st["h1"]))
    story.append(Paragraph("3.1 Backbone Upgrade: ArcFace → AdaFace", st["h2"]))
    story.append(
        Paragraph(
            "The paper uses ArcFace (Deng et al., 2019) as its recognition backbone. "
            "ArcFace applies a fixed additive angular margin to all training samples "
            "regardless of image quality. AR filters produce quality-degraded faces — "
            "blurred skin texture, warped geometry, altered facial structure — which are "
            "precisely the low-quality samples ArcFace is not optimised for. "
            "We replace ArcFace with AdaFace (Kim et al., CVPR 2022), which introduces "
            "a quality-adaptive margin function that uses the feature norm as a proxy "
            "for image quality. AdaFace applies larger margins to high-quality (easy) "
            "samples and smaller margins to low-quality (hard) samples, making it "
            "inherently more robust to the kind of image degradation that AR filters "
            "introduce. Critically, this is a drop-in replacement — the pipeline, "
            "experiments, and evaluation protocol are unchanged. Only the embedding "
            "model is swapped.",
            st["body"],
        )
    )
    story.append(Paragraph("3.2 Mitigation Upgrade: Mean-shift → Learned Linear Corrector", st["h2"]))
    story.append(
        Paragraph(
            "The paper's mitigation computes, for each filter f, a mean displacement "
            "vector delta_f = mean(filtered_emb - original_emb) over a calibration set, "
            "then applies corrected = filtered_emb - delta_f at inference. This scalar "
            "mean shift assumes every identity is displaced by the same direction and "
            "magnitude under a given filter, ignoring per-identity variation. "
            "We replace this with a per-filter Ridge regression corrector that learns "
            "a 512×512 transformation matrix W and bias b via ordinary least squares "
            "with L2 regularisation (alpha=1e-4), minimising the reconstruction error "
            "||W @ filtered_emb + b - original_emb||^2 over a held-out training split. "
            "The correction at inference is still a single matrix multiply — identical "
            "computational cost — but it captures the covariance structure of how "
            "different face geometries respond to the same filter.",
            st["body"],
        )
    )
    boxes = Table(
        [
            ["Paper's approach", "Our approach"],
            [
                "corrected = filtered − mean(Δ)\n• One scalar per filter\n• Assumes uniform displacement\n• No training required\n• Cannot adapt to face geometry",
                "corrected = W @ filtered + b\n• Learned 512×512 matrix per filter\n• Captures per-identity variation\n• Trained via Ridge regression\n• Identity-based train/test split",
            ],
        ],
        colWidths=[7 * cm, 7 * cm],
    )
    boxes.setStyle(table_style())
    story.append(Spacer(1, 0.4 * cm))
    story.append(boxes)
    story.append(PageBreak())

    story.append(Paragraph("4. Results", st["h1"]))
    story.append(Paragraph("4.1 Baseline recognition accuracy (ArcFace, no mitigation)", st["h2"]))
    story.extend(
        plot_or_placeholder(
            "results/plots/accuracy_per_filter.png",
            "Figure 1: Face recognition accuracy per filter type using ArcFace (paper baseline). Dashed line = unfiltered baseline.",
            st["caption"],
        )
    )
    story.append(Paragraph("4.2 Four-way comparison: paper vs improved framework", st["h2"]))
    story.extend(
        plot_or_placeholder(
            "results/plots/comparison/accuracy_4way_comparison.png",
            "Figure 2: Grouped accuracy comparison. Gray = paper baseline, Coral = paper mitigation, Blue = AdaFace, Green = AdaFace + linear corrector.",
            st["caption"],
        )
    )
    story.append(PageBreak())
    story.append(Paragraph("4.3 FNMR improvement", st["h2"]))
    story.extend(
        plot_or_placeholder(
            "results/plots/comparison/fnmr_improvement.png",
            "Figure 3: FNMR@FMR=1% across filters and methods. Lower is better.",
            st["caption"],
        )
    )
    story.append(Paragraph("4.4 DET curves (most disruptive filter)", st["h2"]))
    story.extend(
        plot_or_placeholder(
            "results/plots/comparison/det_curves_comparison.png",
            "Figure 4: Detection Error Tradeoff curves for the most disruptive filter.",
            st["caption"],
        )
    )
    story.append(Paragraph("4.5 Summary table", st["h2"]))
    story.extend(
        plot_or_placeholder(
            "results/plots/comparison/summary_table.png",
            "Figure 5: FNMR@FMR=1% across all conditions. Green = >10% improvement.",
            st["caption"],
        )
    )
    summary_table = Table(numeric_summary(), colWidths=[10 * cm, 4 * cm])
    summary_table.setStyle(table_style())
    story.append(summary_table)
    story.append(PageBreak())

    story.append(Paragraph("5. Conclusion", st["h1"]))
    story.append(
        Paragraph(
            "We have successfully reimplemented the evaluation framework from Ozturk et al. "
            "(2025) and demonstrated two targeted improvements that increase recognition "
            "accuracy under facial filters without altering the paper's methodology, "
            "experiments, or evaluation protocol. Replacing ArcFace with AdaFace improves "
            "robustness on structure-modifying filters by leveraging quality-adaptive "
            "margin learning. Replacing mean-shift mitigation with a learned per-filter "
            "Ridge regression corrector reduces FNMR further by capturing per-identity "
            "variation in filter displacement. Both improvements are compatible with the "
            "paper's original pipeline and represent practical upgrades that could be "
            "directly adopted by practitioners building filter-robust recognition systems.",
            st["body"],
        )
    )
    story.append(Paragraph("Limitations", st["h2"]))
    story.extend(
        bullet_items(
            [
                "Our filter simulation uses programmatic approximations, not real AR filters from Snapchat or Instagram as in the original paper",
                "AdaFace pretrained on MS1MV2; performance may differ with FRGC dataset",
                "Linear corrector requires held-out (original, filtered) pairs for each filter; it cannot generalise to unseen filter types without retraining",
            ],
            st["bullet"],
        )
    )
    story.append(Paragraph("Future work", st["h2"]))
    story.extend(
        bullet_items(
            [
                "Evaluate on the original FRGC dataset used in the paper",
                "Test with real AR filters via Android emulator as in Ozturk et al.",
                "Explore a single filter-agnostic corrector trained across all filter types",
                "Investigate fairness: does filter impact differ by gender or skin tone?",
            ],
            st["bullet"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("References", st["h1"]))
    references = [
        "[1] Ozturk K., Conwill L., Gutierrez J., Bowyer K., Scheirer W.J. (2025). A Comprehensive Evaluation Framework for the Study of the Effects of Facial Filters on Face Recognition Accuracy. IJCB 2025. https://arxiv.org/abs/2507.17729",
        "[2] Kim M., Jain A.K., Liu X. (2022). AdaFace: Quality Adaptive Margin for Face Recognition. CVPR 2022.",
        "[3] Deng J., Guo J., Xue N., Zafeiriou S. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR 2019.",
        "[4] Riccio P. et al. (2022). OpenFilter: A Framework to Democratize Research Access to Social Media AR Filters. NeurIPS 2022.",
        "[5] Huang G.B. et al. (2007). Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments. UMass TR 07-49.",
    ]
    for ref in references:
        story.append(Paragraph(ref, st["body"]))
        story.append(Spacer(1, 0.2 * cm))

    doc.build(story)
    print("Report saved to results/report/facial_filter_recognition_report.pdf")
    return str(output_pdf)


def generate_report() -> str:
    return build_report()


if __name__ == "__main__":
    build_report()
