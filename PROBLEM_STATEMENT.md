# Problem statement and organiser-provided materials

## The task

The IIIT Bangalore Deepfake ML Hackathon asks participants to build a binary image classifier that distinguishes **real face images (label 0)** from **synthetic/manipulated deepfake face images (label 1)**. It must generalize to different generation methods and altered image quality, then generate one model-derived prediction for every unlabelled test image.

The primary metric is **AUC-ROC of the fake probability**, not classification accuracy. The organisers use private held-out labels and a sealed evaluation: no intermediate leaderboard scores are published.

This document summarizes the supplied competition documents. The unchanged originals below are the source of truth. Our implementation plan, completed work and measured validation scores are in [the project README](README.md).

## Original files supplied by the organisers

All four files below were copied byte-for-byte from the supplied archive, `drive-download-20261101T063453Z-1-001.zip`.

| File | Purpose |
|---|---|
| [Participant README](organizer_materials/README.md) | Original problem description, dataset layout, label mapping, rules, required deliverables and inference interface. |
| [Evaluation metric guide](organizer_materials/evaluation_metric.md) | Original scoring formula, efficiency tiers, robustness criteria, documentation rubric, bonuses and penalties. |
| [Inference starter template](organizer_materials/inference_template.py) | Original example to be customized; it is not our final model or production inference script. |
| [Sample submission](organizer_materials/sample_submission.csv) | Original header/format example with dummy rows; it is not a valid completed submission. |

### Known issue in the supplied starter template

The original `inference_template.py` does not compile: inside `main()`, `BATCH_SIZE` and `THRESHOLD` are used before their `global` declaration. It also contains TODO sections and an example EfficientNet-B4, rather than the model trained in this project. It is preserved unchanged for provenance. **Use [code/inference_script.py](code/inference_script.py) with our trained weights for actual inference.**

The original sample CSV repeats a placeholder image ID and uses dummy confidence values. Preserve it as a format reference; do not submit it as predictions.

## Provided data

- Images are aligned/face-centred **224 x 224 JPEG crops**.
- The local verified training set contains **7,000** labelled images: **3,478 real** and **3,522 fake**. These counts were verified from `train_metadata.csv`; the original documents describe the structure and label convention.
- The local verified test set contains **3,000** unlabelled images, in one flat `test/images/` directory with UUID filenames.
- `train_metadata.csv` maps each training `image_id` to its label; `image_id` is the filename without the `.jpg` extension.
- The organisers describe intentional compression/noise and possible annotation ambiguity, and warn that models relying on JPEG/noise shortcuts may generalize poorly.

The supplied outer archive also contains `hackathon_dataset.zip`. The images and training metadata are not stored in this repository; download them from the organisers and pass their location with `--data_root` and `--test_dir`.

The dataset contains `train/real/`, `train/fake/`, `train_metadata.csv`, and `test/images/`. No official test labels are supplied.

## Scoring and constraints

| Criterion | Maximum points | What the organisers evaluate |
|---|---:|---|
| Accuracy component | 50 | Private-test AUC-ROC computed from the `confidence` column. |
| Efficiency | 20 | Full-test inference runtime/memory on an NVIDIA T4. |
| Generalization and robustness | 20 | Weighted AUC on undisclosed altered test variants. |
| Documentation | 10 | Problem/rationale, architecture, evidence for hyperparameters, training pipeline, limitations and reproducibility. |

Accuracy component formula from the evaluation guide:

```text
max(0, (AUC - 0.5) / 0.5) * 50
```

Our validation AUC must not be treated as the organisers' private-test AUC or an awarded competition score.

Efficiency tiers in the supplied evaluation guide:

| Full-test T4 inference time | Points |
|---|---:|
| Less than 5 minutes | 20 |
| 5-10 minutes | 15 |
| 10-20 minutes | 10 |
| 20-40 minutes | 5 |
| More than 40 minutes or out of memory | 0 |

The participant README also states a 60-minute maximum inference time. These are different thresholds: 60 minutes is the stated runtime requirement, while the efficiency guide awards zero above 40 minutes. The project's under-3-minute goal was an engineering target, not an extra organiser rule. The completed RTX 4060 Ti timing does not verify cold T4 timing.

Framework requirements: **Python 3.10**, **PyTorch 2.x or TensorFlow 2.x**, and model size **under the stated 5 GB maximum**. Optional bonuses cover interpretability visualizations (+2), a documented diverse ensemble (+2), and both PyTorch/TensorFlow export (+1). No bonus or final competition points are claimed here.

## Main rules

- Train/fine-tune only on the provided training data; no external training datasets or deepfake-specific pretrained detectors.
- ImageNet-pretrained backbones and augmentation of provided training images are permitted.
- Test images must never be used for training, including unsupervised training. Our project additionally kept test images out of model selection, threshold fitting and normalization statistics.
- Predictions must come from the model. Hardcoded predictions, plagiarism and other integrity violations can trigger disqualification.
- One submission per team; the latest upload before the organisers' deadline is evaluated.
- The organisers rerun the submitted inference script. Hardcoded paths can fail, and failure to reproduce submitted predictions carries a penalty.

See the original documents for the complete rules and penalty wording.

## Required inference interface and output

```powershell
python inference_script.py --test_dir ./test/images/ --weights ./model_weights.pt --output predictions.csv
```

In this Desktop source layout, run `python code/inference_script.py` with those same arguments.

The output must have exactly this header:

```csv
image_id,label,confidence
```

- One row for every test image; IDs must match the filenames without extensions.
- `label`: integer 0 for real or 1 for fake.
- `confidence`: a number in [0, 1], meaning probability of **fake**, regardless of the predicted label.
- The supplied sample is illustrative only; missing IDs can be penalized.

## Official delivery checklist

The supplied participant README requests a single Google Drive folder shared as **Anyone with the link can view**, containing:

1. Model-generated `predictions.csv`.
2. Trained model weights.
3. The required `inference_script.py`.
4. `requirements.txt` or `environment.yml`.
5. Source code, via a repository link or source archive with README.
6. A technical report, **minimum four pages** according to the supplied participant README.

Training curves/logs and attention/Grad-CAM visualizations are optional. The project prepared matching folder/ZIP deliverables and both a four-page report and a two-page summary. The supplied originals above request the Drive folder; the extra ZIP and short summary are project delivery choices, not additional requirements from these two official documents. The team name, organiser channel and any later event-specific clarification still need to be applied before official submission. No deadline or public submission link is invented here.
