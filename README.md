# Khilona Color Detector 🎨

A lightweight computer vision app that classifies the dominant color of a toy (khilona) from a photo or live camera capture — **blue**, **purple**, or **yellow** — using a custom dual-branch Keras CNN, served through Streamlit.

**Live app:** deployed on Streamlit Community Cloud (see [Deployment](#deployment))

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Design Notes](#model-design-notes)
- [Getting Started](#getting-started)
- [Training](#training)
- [Deployment](#deployment)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)

---

## Overview

This project takes a photo (camera or upload) of a toy and predicts its dominant color class, along with a confidence score and a "belt" label (a lightweight difficulty/tier tag mapped from color). The entire pipeline — model, inference, and UI — lives in a single Python codebase with no separate backend service; Streamlit handles both.

| | |
|---|---|
| **Classes** | `blue`, `purple`, `yellow` |
| **Input size** | 64×64 RGB |
| **Model size** | ~23K params (~91 KB), <150 KB on disk |
| **Test accuracy** | ~92% (see [Model Design Notes](#model-design-notes) for why this number moved) |
| **Inference** | CPU-only, single forward pass, no GPU required |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        streamlit_app.py                      │
│  ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐  │
│  │ st.camera_    │   │  Image loader /   │   │  Result UI   │  │
│  │ input /       │──▶│  RGBA→white       │──▶│  (swatch,    │  │
│  │ file_uploader │   │  compositor       │   │  belt, conf) │  │
│  └──────────────┘   └────────┬──────────┘   └──────────────┘  │
│                               │                                │
│                     ┌─────────▼─────────┐                     │
│                     │ @st.cache_resource │                     │
│                     │  loaded Keras model │                     │
│                     └─────────┬─────────┘                     │
└───────────────────────────────┼────────────────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   models/khilona_model    │
                    │   .keras + model_meta.json│
                    └────────────────────────────┘
```

### Model: Dual Global + Local Color Network

Rather than a conventional deep CNN, the model is a small **dual-branch fusion network** purpose-built for color classification (not general object recognition):

```
Input (64×64×3)
   │
   ├──▶ GlobalAveragePooling2D ──┐
   ├──▶ GlobalMaxPooling2D ──────┤   "Global color statistics" branch
   │                              │   (mean/max RGB across the whole image —
   │                              │    a strong, cheap signal for a color-ID task)
   │
   └──▶ Conv2D(16) → MaxPool → Conv2D(32) → GlobalAveragePooling2D
                              │
                              │   "Local spatial features" branch
                              │   (captures shape/texture context lightly,
                              │    without a deep backbone)
        Concatenate(3 branches, 38-d)
                  │
            Dense(64, relu) → Dropout(0.2)
                  │
            Dense(3, softmax)
```

**Why this shape, not a standard CNN or transfer-learning backbone:**
- The task is *color* classification, not object recognition — a full ImageNet-pretrained backbone (ResNet/MobileNet) is overkill, slow to load, and pulls in texture/shape biases irrelevant to color.
- Global average/max pooling **directly on the raw input** gives the network an explicit, unfiltered view of overall color composition — the single most informative feature for this task.
- The small conv branch adds just enough spatial context to avoid being fooled by, e.g., a colorful logo patch dominating the global average.
- Total trainable parameters: **7,779** (~30 KB). The model trains in seconds on CPU and loads instantly in a Streamlit Cloud free-tier container.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Model framework | **TensorFlow / Keras** (Functional API) | Multi-branch architecture needs the Functional API, not `Sequential`; Keras' `.keras` format is portable and small. |
| Frontend + serving | **Streamlit** | Single-file UI + inference, free hosting on Streamlit Community Cloud, native camera input (`st.camera_input`) with zero extra JS. |
| Image I/O | **Pillow** + **pillow-heif** | Standard image decode/resize; `pillow-heif` adds iPhone `.HEIC`/`.HEIF` support, common in the training photos. |
| Data pipeline | **NumPy**, **scikit-learn** (`train_test_split`, `classification_report`) | Stratified split + metrics, no need for a heavier pipeline (`tf.data`) at this dataset size. |
| Visualization | **Matplotlib** | Training curve plots (`training_history.png`). |
| Deployment | **Streamlit Community Cloud** | Free, deploys directly from GitHub, no Docker/CI needed. Python version pinned via `.python-version` + dashboard setting (TensorFlow doesn't ship wheels for Cloud's default Python yet). |

---

## Project Structure

```
Khilona(Color detector)/
├── streamlit_app.py        # Entire app: model load, preprocessing, UI
├── train.py                 # Training script — run standalone to reproduce the model
├── models/
│   ├── khilona_model.keras  # Trained weights (~91 KB, committed to git)
│   └── model_meta.json      # Class list, belt mapping, img size, test accuracy
├── src/cleaned_dataset/     # Training images, one folder per class
│   ├── blue/
│   ├── purple/
│   └── yellow/
├── .streamlit/
│   └── config.toml          # Dark theme matching brand colors
├── .python-version           # Pins Python 3.12 for local + (attempted) Cloud builds
├── requirements.txt
└── run_streamlit.bat         # Local launch convenience script (gitignored)
```

---

## Dataset

- **~325 images** total, roughly balanced across 3 classes (108 blue / 109 purple / 108 yellow).
- Mixed sources: real toy photos (many as cutout PNGs with transparency), plus some phone photos of colored objects.
- Images are **not exclusively toys** — some are general colored household objects. This was a deliberate scope decision (kept as-is) rather than a data-cleanliness bug; see [Known Limitations](#known-limitations).
- Split: stratified 80/20 train/test (`train_test_split(..., stratify=y_int)`), fixed seed (`42`) for reproducibility.

---

## Model Design Notes

### A real preprocessing bug, found and fixed

The original training pipeline used `Image.open(path).convert("RGB")` directly. Many dataset images are **RGBA PNGs with transparent backgrounds** (cutout toy photos). Flask/PIL's `.convert("RGB")` does *not* composite transparency onto anything — it just drops the alpha channel and keeps whatever RGB values were stored underneath transparent pixels, which in this dataset was **pure black `(0,0,0)`**.

Measured impact: **81–97% of training images across all three classes** had near-black corner pixels, and **0%** had light/white backgrounds. The model was inadvertently learning "black background → object color" rather than color itself, so it performed well on held-out data (which shared the same artifact) but badly on real-world photos with normal backgrounds.

**Fix** (`train.py::load_rgb`, mirrored in `streamlit_app.py::to_rgb`): detect RGBA/transparency and composite onto a **white** background before resizing, so training and inference see the same distribution real photos actually have.

```python
def load_rgb(path):
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")
```

This is why reported test accuracy dropped from ~97% to ~92% after the fix — the earlier number was inflated by a shortcut the model could no longer exploit. **92% on white-composited data is the trustworthy number.**

### Confidence thresholding

Predictions below **60% confidence** are reported as `"unknown"` rather than forced into the nearest class — avoids confidently-wrong output on out-of-distribution input, at the cost of occasionally declining to answer.

---

## Getting Started

### Prerequisites
- Python 3.12 (pinned via `.python-version` — TensorFlow does not yet ship wheels for newer Python versions)

### Install & Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Or on Windows, double-click `run_streamlit.bat`.

The app opens at `http://localhost:8501`. Use the **Camera** tab to snap a photo or **Upload** to pick an image file.

---

## Training

To retrain the model from scratch (e.g. after adding new images to `src/cleaned_dataset/<class>/`):

```bash
python train.py
```

This will:
1. Load and preprocess all images in `src/cleaned_dataset/` (with the white-composite fix applied)
2. Stratified 80/20 train/test split
3. Train up to 50 epochs with `ReduceLROnPlateau`, checkpointing the best model by validation accuracy
4. Print a full `classification_report` (precision/recall/F1 per class)
5. Save `models/khilona_model.keras` and `models/model_meta.json`
6. Save a training curve plot to `training_history.png`

---

## Deployment

Deployed on **Streamlit Community Cloud**, connected directly to this GitHub repo (`main` branch, entry point `streamlit_app.py`).

Two deploy-specific gotchas hit and fixed during setup, kept here as reference:

1. **Python version mismatch** — Streamlit Cloud's default container used Python 3.14, which has no published TensorFlow wheel. Fixed by pinning Python 3.12 both via a `.python-version` file *and* explicitly in the app's **Settings → General → Python version** dropdown (the file alone was not honored by the platform in practice).
2. **Model not found at deploy time** — `models/*.keras` was originally gitignored (treated as a build artifact), so the deployed container had no weights to load. Since the model is small (~91 KB), it's now committed directly to the repo instead of being excluded.

---

## Known Limitations

- **Composition sensitivity**: the model performs best on single-object, moderately-cropped photos similar to its training distribution. Wide scenes with multiple objects, cluttered backgrounds, or unusual framing are more likely to be misclassified — this is a training-data-coverage limitation, not a bug.
- **Scope is "colored object," not strictly "toy"**: the dataset intentionally includes some non-toy household objects (bottles, phones, stationery) alongside toys, so the model is really a general small-object color classifier rather than a toy-specific one.
- **Three colors only**: no support for red, green, black/white, or multi-color objects yet.

## Roadmap

- Expand dataset with more real-world, multi-background photos to close the composition gap
- Add more color classes
- Optional: second output head for material classification (plastic/fabric/etc.)
