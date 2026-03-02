# liveness-detection

# Face Liveness Detection — Rust + Python (pyo3)

Multi-stage face anti-spoofing pipeline. Rust orchestrates inference across four independent Python model stages via pyo3, fuses their outputs, and produces a three-way liveness decision (Accept / Retry / Reject).

Built as part of the SURT AI ML Engineering Internship (Weeks 13–16: Rust–Python Interfacing).

---

## Architecture

Rust calls each Python stage independently. Python handles model execution only — all fusion logic lives in Rust.

```
Input (image or video)
        │
        ▼
┌───────────────────────────────────────────────┐
│              Python (model.py)                │
│                                               │
│  Stage 1 — Spatial   ResNet18 max softmax     │
│  Stage 2 — Texture   Laplacian variance       │
│  Stage 3 — Motion    Edge-weighted frame diff │
│  Stage 4 — Depth     MiDaS depth variance     │
└───────────────────────────────────────────────┘
        │  four scores [0, 1]
        ▼
┌───────────────────────────────────────────────┐
│              Rust (main.rs)                   │
│                                               │
│  Weighted fusion  →  fused score [0, 1]       │
│  Decision layer   →  Accept / Retry / Reject  │
└───────────────────────────────────────────────┘
```

**Preprocessing:** OpenCV Haar cascade detects and crops the face region (20% margin) before all four stages. Detection rate: 83.9% on evaluation dataset.

**Fusion weights (current):** spatial=0.45, texture=0.30, motion=0.25, depth=0.00  
**Thresholds:** accept ≥ 0.55, retry ≥ 0.38, reject < 0.38  
**Note:** ResNet18 and MiDaS are stub backbones — not fine-tuned for liveness. Depth weight is 0 pending further calibration.

---

## Dataset

Evaluated on the [Anti-Spoofing dataset](https://www.kaggle.com/datasets/tapakah68/anti-spoofing) from Kaggle.

| Folder | Type | Label |
|---|---|---|
| `live_selfie` | Static images of real users | 1 (live) |
| `live_video` | Videos of real users | 1 (live) |
| `printouts` | Static printed photo attacks | 0 (spoof) |
| `cut-out printouts` | Printed photos presented on video | 0 (spoof) |
| `replay` | Screen replay video attacks | 0 (spoof) |

Download the dataset and place it in a `data/` folder at the project root.

---

## Results

Evaluated on 43 samples (18 live, 25 spoof), 36 definitive decisions:

| Metric | Value |
|---|---|
| Definitive accuracy | 83.3% (30/36) |
| FAR (spoof accepted) | 0.0% |
| FRR (live rejected) | 50.0% |

All spoof categories correctly rejected on definitive decisions. High FRR is expected — the spatial backbone (ResNet18/ImageNet) is a stub and will be replaced with a liveness-fine-tuned model in the next phase.

---

## Installation

**Requirements:** Python 3.12, Rust (stable), Cargo

```bash
# Clone repo
git clone <your-repo-url>
cd <repo-name>

# Set up Python venv
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install torch torchvision opencv-python pillow numpy timm scikit-learn matplotlib
```

---

## Running

**Single file:**
```bash
cd rust
cargo run -- path/to/image_or_video.mp4
```

**Batch evaluation:**
```bash
cd rust
cargo run -- --batch ../data 43
```

Output columns: `Spatial | Texture | Motion | Depth | Score | Label | Decision | Outcome`

**ROC curve + AUC analysis:**
```bash
cd python
python roc_curve.py --data ../data --max_n 43
```

Outputs `roc_curve.png` to the `python/` directory and prints AUC + optimal threshold (Youden's J) to terminal.

---

## Project Structure

```
rust_python_inference/
├── rust/
│   └── src/
│       └── main.rs          # Rust orchestrator, fusion, decision layer
├── python/
│   ├── model.py             # Four inference stages + dataset iterator
│   └── roc_curve.py         # ROC/AUC analysis script
├── Cargo.toml
└── Cargo.lock
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pyo3` | Rust–Python bridge |
| `torch` / `torchvision` | ResNet18 spatial backbone |
| `timm` | Required by MiDaS |
| `opencv-python` | Face detection (Haar cascade), video I/O |
| `Pillow` | Image loading |
| `numpy` | Frame processing |
| `scikit-learn` | ROC curve + AUC computation |
| `matplotlib` | ROC curve plotting |
