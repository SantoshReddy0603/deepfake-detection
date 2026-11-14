# Deepfake Detection

Image deepfake detector for 224x224 face crops. The final model is a six-member ensemble of ImageNet-pretrained backbones from five architecture families:

| Member | Family | Pretraining | Params (M) |
|---|---|---|---:|
| EfficientNetV2-S | CNN, MBConv | ImageNet-1k | 20.2 |
| Swin-T | windowed transformer | ImageNet-1k | 27.5 |
| ViT-B/16 | global transformer | ImageNet-21k | 85.8 |
| ConvNeXtV2-T | CNN, FCMAE pretraining | ImageNet-22k | 27.9 |
| ConvNeXt-S (a) | CNN, large kernel | ImageNet-22k | 49.5 |
| ConvNeXt-S (b) | CNN, large kernel | ImageNet-22k | 49.5 |

Each member scores the image and its horizontal flip. The twelve logits are averaged, and a sigmoid of that mean gives the fake probability.

- **Pretraining:** only public ImageNet classification weights are used. No deepfake-specific detector and no web-scale image-text model.
- **Label noise:** the members are trained without 622 images whose labels the models consistently disagree with. The [label-noise study](#label-noise-study) explains why.
- **Report:** `technical_report.pdf` in the output folder covers the same material with figures.

See [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) for the task, scoring, rules, dataset location and submission checklist. The unchanged organiser-provided README, evaluation guide, inference template and sample CSV are in [organizer_materials/](organizer_materials/).

## Main scores

All scores below are out-of-fold (OOF): every one of the 7,000 training images is scored by a model that never trained on it. Folds are 5-fold and group-aware, so near-duplicate families never cross a fold boundary.

| System | Members | OOF AUC | Mean of 16 degradations | Pixelation mean |
|---|---:|---:|---:|---:|
| **Final ensemble (v5)** | 6 | **0.91157** | **0.9102** | **0.8006** |
| Previous ensemble (v4), torchvision ImageNet-1k backbones | 5 | 0.90934 | 0.9074 | 0.7819 |
| All twelve v5 candidate configurations, no selection | 12 | 0.91045 | 0.9091 | 0.7978 |
| v4 backbones trained on all labels (v3) | 5 | 0.90792 | 0.9058 | 0.7715 |
| First submission (ConvNeXt-T + EfficientNet-B0, 3-fold OOF) | 2 | 0.90116 | 0.8982 | not measured |

- **Scored on all labels:** the suspect images are left out of training only. They are still scored with their given labels, so every row is measured on the same 7,000 labels.
- **Gain over v4:** +0.0022 AUC, 95% paired-bootstrap interval [+0.0005, +0.0040]. The gain is positive on all five folds (+0.0014 to +0.0038).
- **Robustness gain over v4:** +0.0028 on the 16-condition mean, interval [+0.0012, +0.0042], and +0.019 on pixelation.
- **Selection check:** the members were chosen on the same OOF predictions they are scored on. When the selection is repeated inside four folds and scored on the fifth, the result is 0.9089, level with v4 (difference -0.0004, interval [-0.0028, +0.0020]). The clean-AUC gain is therefore small and uncertain; the robustness gain is the firmer result.
- **Labels at threshold 0.5:** OOF accuracy 0.876 and F1 0.879. No threshold is fitted.
- **Scope:** these are validation results. The private-test AUC is unknown.

Single members (pooled 5-fold OOF AUC at the saved epoch):

| Member | Epochs | Clean | Mean of 16 degradations | Pixelation mean |
|---|---:|---:|---:|---:|
| EfficientNetV2-S s42 | 11 | 0.9070 | 0.9048 | 0.7534 |
| Swin-T s7 | 10 | 0.9036 | 0.9011 | 0.7450 |
| ViT-B/16 (21k) s42 | 6 | 0.9070 | 0.9056 | 0.7615 |
| ConvNeXtV2-T (22k) s42 | 10 | 0.9062 | 0.9049 | 0.7839 |
| ConvNeXt-S (22k) s7, mosaic 4-24 px | 10 | 0.9065 | 0.9047 | 0.8035 |
| ConvNeXt-S (22k) s42 | 10 | 0.9057 | 0.9033 | 0.7854 |

Ensemble OOF AUC as the members are added in that order: 0.9070, 0.9083, 0.9098, 0.9109, 0.9115, 0.9116.

Trained and left out: DeiT-III-B (0.9037), Swin-S 22k (0.9012), ViT-B with 4-24 px mosaic (0.9051), and the v4 members ConvNeXt-S and ConvNeXt-T with ImageNet-1k weights (0.9056, 0.9062) and ResNet-50 (0.9038). None improved the six-member set on both clean and degraded data.

## Training recipe (`code/train_v2.py`)

- **Augmentation:** JPEG re-compression, blur, noise, down/up-scaling, colour jitter, plus mosaic/pixelation with p = 0.5 and blocks of 4 to 16 px (4 to 24 px for one ConvNeXt-S member).
- **Optimiser and loss:**
  - AdamW with backbone learning rate 1e-4, head learning rate 5e-4 and weight decay 0.05 (ViT-B: 3e-5 and 3e-4);
  - one warm-up epoch, then cosine decay over 12 epochs for EfficientNetV2-S and Swin-T, 10 for the ConvNeXt members and 6 for ViT-B;
  - BCE with label smoothing 0.05;
  - batch 64, fp16 autocast and gradient clipping;
  - EMA weights with decay 0.995.
- **Label noise:** 43 near-duplicate groups (88 images) carry both labels, so at least one label in each is wrong. These groups are left out of training. They stay in validation, so the scores remain comparable.
- **Validation and selection:**
  - Every epoch, the EMA model scores its held-out fold with flip TTA.
  - Members are chosen on the pooled OOF predictions.
  - EfficientNetV2-S and Swin-T, kept from v4, stop at the epoch chosen on OOF (11 and 10 of 12). The four new members run their full schedule.
  - Each chosen member is then refit on all images with the same schedule and saved at that epoch.
- **Ruled out:** averaging neighbouring epochs of the same run gave no gain (0.9076).

## Label-noise study

The question: is the ceiling of about 0.91 caused by wrong labels, or just by hard images? Four checks were run.

The study was run on the previous five-member ensemble (v3 trained on all labels, v4 without the flagged images).

1. **Confident learning.** Per-class thresholds were applied to the v3 OOF probabilities.
   - 653 images are flagged as likely mislabeled: 346 real images that look fake and 307 fake images that look real.
   - That is an estimated noise rate of 9.3%.
   - 31 of the 653 are in the conflict groups already left out of training, so the final members exclude 622 more images.
   - For scale, a perfect detector scored against labels with 10% random flips would reach about 0.900 AUC.
   - This check is consistent with label noise but circular, because the flags come from the same model.
2. **Visual audit.** 50 flagged and 50 control images were labelled blind by eye. On the controls, the eye matched the true labels no better than chance. The fakes are not visibly fake, so a human audit cannot settle the question.
3. **Controlled retraining.** Two members were retrained in full 5-fold CV under three settings: all images, without the flagged images, and without the same number of randomly chosen images (class-matched). All settings were scored on every image with its given label.

   | Two-member ensemble (ConvNeXt-S + EfficientNetV2-S) | OOF AUC |
   |---|---:|
   | All images | 0.90690 |
   | Without 653 flagged images | **0.90892** |
   | Without 653 random images | 0.90346 |

   - **Flagged vs random:** removing the flagged images beats removing random ones by +0.0055 AUC, 95% CI [+0.0026, +0.0083].
   - **Data loss:** random removal costs AUC, as losing data should. Removing the flagged images instead gains AUC, so these images harm training.
   - **All five members:** all five gain when trained without the flagged images, by +0.0011 to +0.0052.
   - **Five-member ensemble:** the gain is smaller, +0.0012, with a CI of [-0.0009, +0.0033], because cleaner members agree more with each other.
4. **Learning curve.** ConvNeXt-T trained on 25%, 50%, 75% and 100% of the training groups scored 0.870, 0.892, 0.900 and 0.902. The curve is flattening: each doubling of data gains about half as much as the previous one. More data of the same kind would add little.

**Conclusion:**
- A meaningful share of the flagged images behave like wrong labels, and training without them helps. This is the strongest evidence that label noise limits the score.
- It does not prove which individual labels are wrong.
- The flags were built from OOF predictions that had seen the other folds' labels, which is a mild leak, so the controlled gain may be slightly optimistic.
- The same noise is presumably in the test labels, so the measured AUC is capped. 0.92 is unlikely to be reachable on these labels.

## Robustness

Every member scored its held-out fold under 16 degradations, plus three harsher pixelation levels. Each degradation is seeded per image, so every model sees the same degraded pixels. The table shows pooled 5-fold OOF AUC on all 7,000 images, with flip TTA.

| Condition | v5 (final) | v4 |
|---|---:|---:|
| clean | 0.9116 | 0.9093 |
| JPEG q30 / q50 / q70 | 0.9101 / 0.9112 / 0.9111 | 0.9070 / 0.9085 / 0.9084 |
| blur r1 / r2 | 0.9110 / 0.9068 | 0.9083 / 0.9034 |
| noise 5 / 10 | 0.9113 / 0.9110 | 0.9083 / 0.9079 |
| downscale 112 / 160 | 0.9103 / 0.9112 | 0.9073 / 0.9084 |
| brightness 0.8 / 1.2 | 0.9116 / 0.9109 | 0.9092 / 0.9088 |
| contrast 0.8 / 1.2 | 0.9115 / 0.9113 | 0.9092 / 0.9094 |
| centre crop 0.8 | 0.9082 | 0.9057 |
| combined (JPEG + blur + noise + downscale) | 0.9042 | 0.8995 |
| **Mean of the 16 conditions above** | **0.9102** | 0.9074 |
| pixelate 8 / 16 / 24 px | 0.8759 / 0.7966 / 0.7295 | 0.8630 / 0.7769 / 0.7059 |

- **Ahead everywhere:** v5 scores higher than v4 on all 19 rows.
- **Worst of the 16:** the combined degradation, which costs 0.007 AUC.
- **Pixelation:** strong pixelation is still the clear weak point, even with mosaic augmentation. At 24 px blocks most facial detail is gone.
- **Full table:** per-member numbers are in `robustness_table.csv` next to the bundle.

## Install and predict

Use Python 3.10 in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python code/inference_script.py --test_dir "C:\path\to\images" --weights "C:\path\to\model_weights.pt" --output predictions.csv
```

- **Output:** columns `image_id,label,confidence`. `confidence` is the fake probability, and `label` is 1 (fake) when confidence >= 0.5.
- **Devices:** CPU is supported; CUDA uses fp16 autocast.
- **Robustness:**
  - Unreadable images get a warning and confidence 0.5.
  - A CUDA out-of-memory error halves the batch size and retries.
  - Inference needs no network or training data.
- **Dependencies:** `timm` builds four of the six backbones. It is listed in `requirements.txt`, and nothing is downloaded at inference.
- **Bundle:** 522 MB, format 3, fp16 weights and plain tensors only.
- **Runtime:**
  - Fresh process, 3,000 test images on an RTX 4060 Ti: 59 s, including model load and CSV writing, with peak CUDA memory of 2.5 GB.
  - Tesla T4 (Kaggle), same command, three fresh runs: 124 s, 111 s and 108 s wall clock, with peak CUDA memory of 2.2 GB. The first run includes the first read of the files from disk. All are inside the 5-minute limit.
  - The T4 output matches the submitted `predictions.csv` on all 3,000 images: no label differences, maximum confidence difference 0.0006.
- **CPU/GPU agreement:** maximum confidence difference 0.0003 and no label flips on 32 test images.

## Grad-CAM

```powershell
python code/gradcam.py --weights model_weights.pt --images some_dir --out gradcam --limit 16
```

For each image the script writes one panel: the input, then one heatmap per member. The fused probability is in the file name. The target layer is the last convolutional stage for the CNNs, the last stage for Swin-T and the token grid of the last block for ViT-B. On the test samples checked, the convolutional members and Swin-T mostly focus on the central face (eyes, nose, mouth). The ViT-B map is more diffuse.

## TensorFlow export

The same ensemble is also delivered as TensorFlow SavedModels, one per member, in `tf_export/`.

```powershell
python -m pip install tensorflow==2.19.1 tf_keras==2.19.0 onnx==1.17.0 onnxsim==0.8.1 onnx2tf==1.29.24 onnxruntime sng4onnx ai-edge-litert psutil
python code/export_tf.py --weights model_weights.pt --out tf_export --check_dir some_dir --check_images 32
python code/tf_inference.py --test_dir some_dir --weights tf_export --output predictions_tf.csv
```

- **Route:** PyTorch to ONNX (opset 17), graph simplification, then `onnx2tf`. Each member is converted in its own process to keep memory low.
- **Parity:** `export_tf.py` compares TensorFlow and PyTorch logits on real images and writes the gaps to `tf_export/manifest.json`. For the submitted export the largest logit gap is 1.0e-05 and the largest probability gap 1.7e-07 on 32 test images; `tf_inference.py` gives the same labels as the PyTorch script on those images.
- **Prediction:** `tf_inference.py` needs only TensorFlow, NumPy and Pillow. The export dependencies are not needed for the PyTorch inference script.

## Reproducing training

```bash
# one fold run, or an all-data refit that saves the chosen epochs
python code/train_v2.py --data_root DATA --split fold0.csv --arch convnext_small_in22k --seed 7 --fold 0 \
    --epochs 10 --mosaic_max 24 --robust --exclude flagged.csv --out runs
python code/train_v2.py --data_root DATA --split fold0.csv --arch convnext_small_in22k --seed 7 --fold all \
    --epochs 10 --save_epochs 10 --mosaic_max 24 --exclude flagged.csv --out runs
# choose members on pooled OOF, then pack the refit members
python code/select_ensemble.py --runs runs --baseline oof_predictions.csv --out chosen.json
python code/select_ensemble.py --runs runs --chosen chosen.json --bundle model_weights.pt
```

Other training options:
- `--exclude flagged.csv`: never train on the listed image ids.
- `--train_frac 0.5`: train on a random fraction of the training groups.
- `--robust`: after the last epoch, score the validation fold under every degradation.

Job lists in `code/gpu/`: `jobs_d.txt` holds the label-noise and robustness runs, `jobs_e.txt` the v4 refits, `jobs_f.txt` the v5 cross-validation runs and `jobs_refit5.txt` the v5 refits.

`code/gpu/cluster.py` ran the training jobs on shared lab GPUs (66 jobs, then 80 for the noise and robustness study, then 67 for v5):

- **Setup:**
  - every job ran in a private work directory marked with an owner file;
  - caches and the Python environment stayed inside that directory;
  - jobs ran as detached queues.
- **Cleanup:** the `cleanup` subcommand stops only our queue processes and deletes only our directory. It ran after training, and every host was checked afterwards.

One fold of ConvNeXt-Small takes about 6 minutes on an RTX 4060 Ti.

## Folder contents

- **Final pipeline:**
  - `code/train_v2.py`: CV and refit;
  - `code/select_ensemble.py`: member selection and bundling;
  - `code/model_def.py`: architectures and bundle format;
  - `code/augment.py`: augmentation;
  - `code/inference_script.py`: prediction;
  - `code/gradcam.py`: heatmaps;
  - `code/export_tf.py` and `code/tf_inference.py`: TensorFlow export and prediction;
  - `code/gpu/`: the shared-GPU runner and its job lists.
- **Earlier ConvNeXt-Tiny + B0 pipeline, still runnable:**
  - `code/run_stage_a.py` and `run_stage_b.py`;
  - `run_cv.py` (also provides the group folds used by `train_v2.py`);
  - `noise_weights.py`, `run_ablation.py` and `compare_ensemble.py`;
  - `run_ensemble_cv.py` and `run_ensemble_refit.py`;
  - `run_refit.py`, `validate_precision.py` and `verify_submission.py`.
- **Other files:**
  - `PROBLEM_STATEMENT.md`;
  - `organizer_materials/`;
  - `requirements.txt` (inference dependencies);
  - `LICENSE`.

Images, weights, predictions and logs are not stored here. The submission files are in the `DeepfakeDetector_v5` folder shared on Google Drive:

- `model_weights.pt` and `predictions.csv` for the 3,000 test images;
- `technical_report.pdf`;
- `ensemble_selection.json` and `robustness_table.csv`;
- `noise_study.json` and `flagged.csv` (the excluded ids);
- `gradcam/` and `tf_export/`.

Earlier ensembles are in `outputs\DeepfakeDetector_v4`, `outputs\DeepfakeDetector_v3` and `outputs\DeepfakeDetector`.

## Limitations

- **Label noise.** The controlled retraining in the [label-noise study](#label-noise-study) shows that the flagged images act like wrong labels. Which individual labels are wrong cannot be verified, because the fakes are not visibly fake.
- **Selection optimism.** Members and the exclusion list were derived from OOF predictions on the same images. The nested check under [Main scores](#main-scores) shows the clean-AUC advantage over v4 is not statistically secure. The exclusion list adds a mild leak, described in the label-noise study.
- **Pixelation.** AUC drops to 0.80 at 16 px blocks and to 0.73 at 24 px blocks.
- **Unknown test variants.** Robustness was measured on our own choice of degradations; the organisers' variants may differ.
