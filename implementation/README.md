# Facial Filter Impact on Face Recognition

Prototype reimplementation of "A Comprehensive Evaluation Framework for the Study of the Effects of Facial Filters on Face Recognition Accuracy" by Kagan Ozturk, Louisa Conwill, Jacob Gutierrez, Kevin Bowyer, and Walter J. Scheirer, University of Notre Dame. Paper: https://arxiv.org/abs/2507.17729

This project builds an end-to-end face verification pipeline on LFW: prepare verification pairs, simulate facial filters, extract face embeddings, measure recognition degradation, apply embedding-space mitigation, compare backbone/correction strategies, and expose the workflow in a Gradio demo dashboard.

## Setup

```bash
cd facial_filter_project
pip install -r requirements.txt
```

The setup script searches `~/Downloads` for an LFW folder or archive. In this workspace it can use `~/Downloads/archive/lfw-deepfunneled/lfw-deepfunneled`. If `pairs.txt` is not already available, it tries to download the standard LFW pairs file and otherwise falls back to local LFW pair CSVs.

## Run the Full Pipeline

```bash
python src/07_run_pipeline.py --subset 300 --models arcface
```

Run the improved-backbone and linear-corrector comparison steps:

```bash
python src/07_run_pipeline.py --upgrade-only --subset 300
```

Regenerate the PDF report only:

```bash
python src/07_run_pipeline.py --report-only
```

Each script can also run independently:

```bash
python src/01_setup_dataset.py --subset 300
python src/02_apply_filters.py --subset 300 --workers 4
python src/03_filter_selection.py --sample-size 50
python src/04_extract_embeddings.py --models arcface --subset 300
python src/04b_extract_embeddings_adaface.py --subset 300
python src/05_evaluate_recognition.py --backbone arcface
python src/05_evaluate_recognition.py --backbone adaface
python src/06_mitigation.py
python src/06b_mitigation_linear.py --backbone both
python src/08_compare_models.py
```

## Run the Demo

```bash
python app/app.py
```

The Gradio app launches at http://127.0.0.1:7860. The current app is a dashboard-style live demo with upload-based verification and real regenerated analytics plots.

## Outputs

- `data/lfw_original/`: prepared LFW identity folders.
- `data/lfw_filtered/<filter>/`: simulated filtered images for blur, brightness, skin smoothing, eye enlargement, face slimming, and warm color tone.
- `data/pairs_genuine.csv` and `data/pairs_impostor.csv`: verification pairs with labels.
- `results/embeddings/`: saved `.npy` embeddings by model, filter, identity, and filename.
- `results/embeddings/embedding_index.csv`: embedding metadata and validity flags.
- `results/metrics/filter_impact_scores.json`: mean embedding distance between original and filtered images.
- `results/metrics/recognition_metrics.json`: accuracy and FNMR metrics per filter.
- `results/metrics/recognition_metrics_adaface.json`: improved-backbone metrics. In this Python 3.13/macOS environment, AdaFace was unavailable and the configured GhostFaceNet fallback was used.
- `results/metrics/mitigation_metrics.json`: filter detector accuracy and corrected recognition metrics.
- `results/mitigation_linear/linear_mitigation_metrics.json`: Ridge linear-corrector metrics for ArcFace and the improved backbone path.
- `results/plots/accuracy_per_filter.png`: accuracy bar chart.
- `results/plots/det_curves.png`: DET curves.
- `results/plots/similarity_heatmap.png`: mean genuine/impostor similarity heatmap.
- `results/plots/mitigation_comparison.png`: before/after correction chart.
- `results/plots/comparison/`: four-way model/mitigation comparison plots used by the dashboard and report.

## Current Submitted Experiment

The May 31 submission uses a runtime-manageable LFW subset:

- 300 genuine pairs + 300 impostor pairs.
- 807 unique pair images filtered under each of 6 filter categories.
- 4,842 generated filtered images.
- ArcFace baseline and improved-backbone embeddings saved as `.npy` arrays.
- All dashboard statistics and comparison plots are computed from saved pair scores, metrics JSON files, and DET CSV files.

## Dataset Description for Submission

Dataset source:

- Public Labeled Faces in the Wild (LFW) dataset.
- Standard LFW `pairs.txt` verification protocol.
- No private images were collected.

Prepared dataset in this project:

- `data/lfw_original/`: 13,233 original LFW images across 5,749 identities.
- `data/lfw_filtered/`: 4,842 generated filtered images for the 807 unique images used by the 600-pair experiment subset.
- `data/pairs_genuine.csv`: 300 same-person pairs.
- `data/pairs_impostor.csv`: 300 different-person pairs.

Preprocessing:

1. Extract LFW into identity folders.
2. Parse standard LFW pairs.
3. Generate six filter categories with OpenCV/PIL.
4. Extract embeddings.
5. Compute cosine-similarity verification metrics and DET curves.

Ethical/legal note:

LFW is a public research benchmark containing face images, so it should be used only for academic evaluation. This project is a course prototype and not a production identity or surveillance system. Generated filtered images are derived from LFW and should be treated under the same research-use expectations.

## What to Submit on May 31

Submit the project folder `facial_filter_project/` as the code implementation. It contains:

- `src/`: setup, filtering, embedding extraction, evaluation, mitigation, comparison, and report scripts.
- `app/app.py`: live Gradio demo dashboard.
- `README.md`: setup, dataset, run commands, outputs, and submission notes.
- `requirements.txt`: reproducible Python dependencies.
- `results/metrics/`: real regenerated JSON/CSV metrics.
- `results/plots/`: real regenerated evaluation and comparison plots.
- `results/mitigation_linear/`: trained Ridge correctors and linear mitigation metrics.

Submit the dataset as either the `data/` folder or a dataset zip/link containing:

- `data/lfw_original/`
- `data/lfw_filtered/`
- `data/pairs.txt`
- `data/pairs_genuine.csv`
- `data/pairs_impostor.csv`

Do not submit local cache folders such as `.deepface/`, `.matplotlib/`, or `__pycache__/` unless your portal specifically asks for pretrained weights/cache files.

## Prototype Note

This is a prototype reimplementation using programmatic filter simulation. The original paper used real AR filters from Instagram, Snapchat, Meitu, and Pitu applied via Android emulator. Our filter implementations approximate the same transformation categories: beautification, color grading, softening, and structural changes.
