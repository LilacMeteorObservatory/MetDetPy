# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MetDetPy is a Python-based meteor detection system that detects meteors from videos and images. It uses frame-differencing and Hough line detection (M3Detector) or deep learning models (MLDetector/DLDet) as the primary detection method, with optional YOLO-based recheck for classification and false-positive reduction.

## Running the Tools

```sh
# Video detection
python MetDetPy.py video.mp4
python MetDetPy.py video.mp4 --cfg config/m3det_normal.json --save-path results.json

# Image detection
python MetDetPhoto.py ./images --save-path results.json

# Clip/stack toolkit (post-processing)
python ClipToolkit.py results.json --mode video --save-path ./output

# Evaluation (requires annotation file and test video in test/)
python evaluate.py test/20220413Red.mp4 test/20220413_annotation.json

# Package as executable (requires nuitka)
python make_package.py
```

## Dependencies

```sh
pip install -r requirements.txt
```

Key deps: numpy, opencv-python, tqdm, onnxruntime, av (PyAV), dacite, pyexiv2, rawpy.

## Architecture

The detection pipeline flows linearly:

```
VideoFile → VideoLoader → Detector → Collector → Exporter → MDRF JSON
```

### Core Components (`MetLib/`)

- **VideoLoader** (`videoloader.py`): Decodes video, applies masks, resizes, estimates exposure time, merges frames. Variants: `VanillaVideoLoader` (sync), `ThreadVideoLoader` (threaded). Uses a **VideoWrapper** (`videowrapper.py`) abstraction over OpenCV or PyAV backends.

- **Detector** (`Detector.py`): Detects candidate meteor responses within a sliding time window. Hierarchy:
  - `BaseDetector` (ABC)
  - `LineDetector` → `ClassicDetector`, `M3Detector` (frame-differencing + Hough lines, grayscale)
  - `MLDetector` (YOLO-based detection, requires color frames)
  - `BrightnessDetector` (whole-frame brightness events)

- **Collector** (`collector.py`): Aggregates per-frame "responses" into motion sequences (tracks), applies motion-based filtering (speed, duration, direction), then optionally invokes a YOLO recheck model for classification.

- **Model** (`model.py`): ONNX Runtime inference wrapper for YOLO models. Handles multi-scale prediction, NMS, and provider selection (CPU/CUDA/DirectML/CoreML).

### Supporting Components

- **metstruct** (`metstruct.py`): All dataclass definitions — config structs (`MainDetectCfg`, `BinaryCfg`, `ModelCfg`, etc.), runtime params, and the output format (`MDRF`, `SingleMDRecord`). Uses `dacite` for JSON→dataclass deserialization.
- **metlog** (`metlog.py`): Async logging system with backend/frontend modes.
- **metvisu** (`metvisu.py`): OpenCV-based visualization for debug mode.
- **utils** (`utils.py`): Constants (`VERSION`), path resolution, math helpers, sliding window, EMA, coordinate transforms.
- **stacker** (`stacker.py`): Frame stacking algorithms (max, MFNR, denoise).
- **imgproc** (`imgproc.py`): Image transform pipeline.

### Entry Points

| File | Purpose |
|------|---------|
| `MetDetPy.py` | Video meteor detection (main tool) |
| `MetDetPhoto.py` | Image/photo meteor detection |
| `ClipToolkit.py` | Post-detection clipping, stacking, export |
| `evaluate.py` | Evaluation against ground-truth annotations |
| `make_package.py` | Nuitka packaging to standalone executable |

### Configuration

JSON config files in `config/` define the full pipeline (loader, detector, collector settings). The default is `config/m3det_normal.json`. Config is deserialized into `MainDetectCfg` via dacite.

### Output Format (MDRF)

Detection results use the MetDetPy Detection Result Format (MDRF) — a JSON structure containing video metadata, config used, and a list of `SingleMDRecord` entries with bounding boxes, timestamps, confidence scores, and category labels.

## Key Conventions

- The project is bilingual (Chinese/English) in comments and docs. Code identifiers are in English.
- Python 3.9+ features are used (type hints with `list[...]`, `tuple[...]`).
- No formal test suite — testing is done via `evaluate.py` against annotated test videos.
- Resource files (weights, class names, clip config) are resolved relative to the project root via `relative2abs_path()` in utils, overridable with `--resource-dir` or `METDET_RESOURCE_DIR` env var.
- Class names are defined in `global/class_name.txt` and loaded lazily.
- ONNX model weights live in `weights/` (tracked via Git LFS).
