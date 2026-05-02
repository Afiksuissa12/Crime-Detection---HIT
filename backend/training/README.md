# Violence Detection — Training Module

## Goal

Train a binary video classifier to distinguish `violence` from `no_violence` clips.
The trained checkpoint is loaded by the running Docker backend for real-time inference.

---

## Step 0 — Download RWF-2000

RWF-2000 is not included in this repository and must be downloaded manually.

**Source:** Kaggle — search for "RWF-2000 Dataset" or use the original paper link:
> *RWF-2000: An Open Large Scale Video Database for Violence Detection*
> Ming-Ching Chang, Guang-Yu Nie et al.

The dataset contains **2,000 real-world surveillance video clips**:
- 1,000 clips labelled `Fight` (violence)
- 1,000 clips labelled `NonFight` (no violence)

After downloading, extract the archive and sort the clips into the folder
structure described in Step 1.

---

## Step 1 — Place Dataset Files

The folder scaffold already exists in the repository (`.gitkeep` files).
Video files are excluded from git (see `datasets/.gitignore`).

Copy or move your clips into the following layout:

```
backend/training/datasets/rwf2000/
├── train/
│   ├── violence/        ← paste violence clips here       (~800 files)
│   └── no_violence/     ← paste non-violence clips here   (~800 files)
├── val/
│   ├── violence/                                          (~100 files)
│   └── no_violence/                                       (~100 files)
└── test/
    ├── violence/                                          (~100 files)
    └── no_violence/                                       (~100 files)
```

**Recommended split for RWF-2000 (80 / 10 / 10):**

| Split | violence | no_violence | Total |
|-------|----------|-------------|-------|
| train | 800      | 800         | 1,600 |
| val   | 100      | 100         | 200   |
| test  | 100      | 100         | 200   |

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`

---

## Step 2 — Install Dependencies

Run from `backend/training/`:

```bash
cd backend/training
pip install torch torchvision opencv-python scikit-learn
```

GPU is strongly recommended.
- CPU: ~2–4 hours for 20 epochs on RWF-2000
- Single GPU (e.g. RTX 3060): ~15–30 minutes

---

## Step 3 — Validate the Dataset

Before training, verify every video file is readable and the folder structure
is correct:

```bash
python scripts/prepare_video_dataset.py --data_root datasets/rwf2000
```

Expected output (example):

```
         split           class           total              ok             bad
  -----------------------------------------------------------------------------
         train        violence            800             800               0
         train    no_violence            800             800               0
           val        violence            100             100               0
           val    no_violence            100             100               0
          test        violence            100             100               0
          test    no_violence            100             100               0
  -----------------------------------------------------------------------------
         TOTAL                          2000            2000               0

  [OK]  All 2000 clips verified successfully.
```

Fix any reported bad files before continuing.

---

## Step 4 — Train

```bash
python scripts/train_violence_baseline.py \
    --data_root  datasets/rwf2000 \
    --epochs     20               \
    --batch_size 8                \
    --lr         1e-4             \
    --output_dir models/
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `datasets/rwf2000` | Root of the split folders |
| `--epochs` | 20 | Training epochs |
| `--batch_size` | 8 | Clips per batch (reduce to 4 if out of memory) |
| `--lr` | 1e-4 | Adam learning rate |
| `--output_dir` | `models/` | Where to save checkpoints |
| `--no_pretrain` | off | Train from scratch — not recommended |

The best checkpoint (by validation F1) is saved automatically to:

```
backend/training/models/violence_baseline_model.pth
```

Training log example:

```
Epoch 01/20  train_loss=0.6821  val_loss=0.5903  val_acc=0.7250  val_f1=0.7201  (48.3s)
Epoch 02/20  train_loss=0.5412  val_loss=0.4871  val_acc=0.7900  val_f1=0.7888  (47.1s)
...
  [BEST] New best checkpoint (F1=0.8812) -> models/violence_baseline_model.pth
```

---

## Step 5 — Evaluate

Run evaluation on the held-out test split:

```bash
python scripts/evaluate_violence_baseline.py \
    --checkpoint models/violence_baseline_model.pth \
    --data_root  datasets/rwf2000                   \
    --split      test
```

Example output:

```
=== Evaluation Results ===
  Accuracy  : 0.8850
  Precision : 0.8863  (weighted)
  Recall    : 0.8850  (weighted)
  F1 Score  : 0.8851  (weighted)

  Confusion Matrix  (rows = true label, cols = predicted label)
                    no_violence       violence
       no_violence           91              9
          violence            14            86

  [OK]      F1 0.8851 meets the production target (>= 0.85)
```

**Production target: F1 ≥ 0.85 on the test split.**

Recall is prioritised over precision — missing a violent event (false negative)
is more costly than a false alarm.

---

## Step 6 — Deploy to Docker Backend

The Docker backend loads the model from the bind-mounted `data/` folder.
Copy the trained checkpoint there — no Docker rebuild required:

```bash
# From backend/training/
cp models/violence_baseline_model.pth ../data/models/violence_baseline_model.pth
```

The model is now available at this path inside the running container:

```
/app/data/models/violence_baseline_model.pth
```

The service `backend/app/services/real_violence_detection_service.py` checks
for this file on every container startup.  Once the file is present, it loads
the model automatically and logs:

```
INFO  ViolenceBaselineModel loaded — epoch=18  val_acc=0.8900  val_f1=0.8851  device=cpu
```

If the file is missing, the service logs a warning and returns a safe fallback
(`"error": "Violence model not trained yet"`) instead of crashing.

---

## Evaluation Metrics Reference

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| **Accuracy** | correct / total | Overall correctness across all clips |
| **Precision** | TP / (TP + FP) | Of all predicted violence clips, how many were truly violent |
| **Recall** | TP / (TP + FN) | Of all actual violence clips, how many were detected |
| **F1** | 2 × P × R / (P + R) | Harmonic mean of precision and recall |
| **Confusion matrix** | — | Full breakdown of TP / TN / FP / FN |

---

## Model Architecture

**ResNet-18 + temporal mean pooling** (ImageNet pretrained)

Each video is decoded to **16 equally-spaced frames** (224 × 224 px).
The shared ResNet-18 backbone extracts a 512-d feature vector from each frame
independently.  Frame vectors are averaged across the temporal dimension
(mean pooling) and fed into a binary linear classifier.

This "frame-averaging" baseline is deliberately simpler than a full I3D model:
it trains faster, requires no 3D convolutions, and performs competitively on
small datasets like RWF-2000.

The `train_i3d_violence.py` / `evaluate_i3d_violence.py` scripts in this folder
implement the full R3D-18 (I3D proxy) approach for when higher accuracy is needed.
