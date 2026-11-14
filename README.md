# FloodNet-ML-project
# Aerial Drone Standing-Water Classification

A computer-vision project that classifies aerial drone imagery as standing water or dry ground. The workflow uses FloodNet imagery and semantic masks to derive image-level labels, compare a baseline neural network with transfer learning, and evaluate performance on a held-out split.

> Project scope: This is an image-classification prototype for experimentation—not a validated flood-warning or emergency-response system. Project-reported results are from notebook runs and should be treated as experimental findings rather than deployment-grade performance claims.

[Open the notebook in Google Colab](https://colab.research.google.com/github/OnyxxSan/FloodNet-ML-project/blob/main/FloodNet_ML_Project.ipynb) · [View the notebook on GitHub](FloodNet_ML_Project.ipynb)

## Results at a glance

| Model | Accuracy | Precision | Standing-water recall |
|---|---:|---:|---:|
| Custom 4-layer CNN baseline | 76% | 85% | 65% |
| MobileNetV2 transfer learning | 81% validation accuracy | — | 82% |

MobileNetV2 improved standing-water recall by 17 percentage points over the baseline and reduced training epochs by 66%, according to the project runs.

> Metric reporting note: The values above are project-reported results, not independently reproduced here. The split used for each metric, class-level metric definitions, decision threshold, and held-out test evaluation should be reported consistently before claiming model performance.

## Why this project

Missing standing water in aerial imagery can matter for post-flood assessment. This project explores whether image classification can flag scenes containing standing water, using semantic masks to convert per-pixel labels into an image-level target. The work is intended as a structured experiment and a portfolio-ready data-science project, not a fielded flood-monitoring system.

## Project workflow

1. Explore the data — inspect FloodNet image/mask pairs and dataset organization.
2. Create image-level labels — analyze semantic masks and assign `standing_water` when the water-pixel fraction exceeds a configurable threshold (reported as 0.5%); otherwise assign `dry_ground`.
3. Train a baseline — build and evaluate a custom four-layer CNN.
4. Apply transfer learning — fine-tune or adapt MobileNetV2 and compare performance with the baseline.
5. Save artifacts — configure model checkpoints and outputs to persist in Google Drive.
6. Run inference — load saved weights and visualize a single-image prediction with its confidence score and overlay.

## Dataset

The notebook uses the FloodNet Supervised v1.0 dataset, with paired original images and semantic-label images. The notebook output shows the following split sizes:

| Split | Image/mask pairs |
|---|---:|
| Train | 1,445 |
| Validation | 450 |
| Test | 448 |

The dataset is not included in this repository. Obtain it from its official source and review its terms of use before downloading or redistributing it.

Expected source layout:

```text
dataset/
└── FloodNet-Supervised_v1.0/
    ├── train/
    │   ├── train-org-img/
    │   └── train-label-img/
    ├── val/
    │   ├── val-org-img/
    │   └── val-label-img/
    └── test/
        ├── test-org-img/
        └── test-label-img/
```

## Reproducible setup

This repository now includes a minimal reproducible environment for local work:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The environment includes the core packages for TensorFlow, NumPy, pandas, scikit-learn, matplotlib, seaborn, Pillow, and Jupyter-compatible tooling.

## Run the notebook

### Google Colab

1. Open the notebook using the Colab link above.
2. Add the FloodNet data to your Google Drive in the location expected by the notebook, or update the data-path settings.
3. Run the notebook cells in order. When prompted, authorize Google Drive access.
4. Review the generated labels and image/mask pairs before training.
5. Checkpoint and output paths are configured in the notebook; change them if your Drive folder differs.

### Local environment

The notebook currently uses Google Colab-specific Drive integration. To run it locally, install the project dependencies, remove or replace the `google.colab` Drive-mount cell, and configure local paths.

## Methods and evaluation

The classification labels are derived from semantic masks using a water-pixel proportion threshold. This converts a pixel-level annotation into an image-level target; images near the threshold may be ambiguous, so the threshold should be documented and monitored.

The repository includes reusable helper functions for label generation and evaluation metrics so the project is easier to interpret and reproduce:

```python
from src.labeling import compute_water_fraction, build_image_label
from src.evaluate import summarize_classification
```

A recruiter-ready evaluation should report all of the following on the final held-out test split:

- Per-class precision, recall, and F1-score
- Confusion matrix and class support
- Precision–recall curve or average precision for the standing-water class
- The classification threshold and how it was selected
- Model input size, preprocessing, random seed, and training configuration
- Clear reporting of validation and test performance separately

Because missed water detections may be more consequential than false alarms in some settings, accuracy alone is not sufficient. Compare models using the standing-water class recall and precision, not only overall accuracy.

## Repository contents

```text
.
├── FloodNet_ML_Project.ipynb   # Data exploration, training, and inference workflow
├── README.md                   # Project overview and setup instructions
├── requirements.txt            # Reproducible Python environment
├── src/
│   ├── __init__.py
│   ├── labeling.py             # Water-fraction and image-label utilities
│   └── evaluate.py             # Classification metric helpers
├── tests/
│   └── test_labeling.py        # Basic checks for label-generation logic
└── reports/                    # Optional output directory for metrics and figures
```

## Data quality and portfolio improvements

The project now includes a clearer structure for reproducibility and evaluation. In a recruiter-ready version, the key next steps are:

- Publish a reproducible environment file with the tested Python, TensorFlow, and supporting-library versions.
- Record class counts after mask-to-image labeling and verify image/mask ID matching and split integrity.
- Add a final held-out test evaluation, per-class metrics, confusion matrix, and precision–recall curve.
- Save the threshold, preprocessing settings, random seed, model configuration, and evaluation results with each run.
- Add representative prediction examples, including false positives and false negatives.
- Treat predictions as decision support only; performance on this dataset does not establish reliability on new locations, sensors, weather, or flood events.

## Limitations and next steps

- Publish a reproducible environment file with the tested Python, TensorFlow, and supporting-library versions.
- Record class counts after mask-to-image labeling and verify image/mask ID matching and split integrity.
- Add a final held-out test evaluation, per-class metrics, confusion matrix, and precision–recall curve.
- Save the threshold, preprocessing settings, random seed, model configuration, and evaluation results with each run.
- Add representative prediction examples, including false positives and false negatives.
- If edge use is a goal, measure model size, inference latency, and memory use on a named target device. The project’s edge-deployment motivation is not itself evidence of on-device benchmarking.
- Treat predictions as decision support only; performance on this dataset does not establish reliability on new locations, sensors, weather, or flood events.

## Acknowledgments

FloodNet dataset creators and maintainers. Dataset citation:

@ARTICLE{9460988,
  author={Rahnemoonfar, Maryam and Chowdhury, Tashnim and Sarkar, Argho and Varshney, Debvrat and Yari, Masoud and Murphy, Robin Roberson},
  journal={IEEE Access},
  title={FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding},
  year={2021},
  volume={9},
  pages={89644-89654},
  doi={10.1109/ACCESS.2021.3090981}
}



