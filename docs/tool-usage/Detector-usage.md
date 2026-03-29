# Detection Tools User Guide

<center> Language: <b>English</b> | <a href="./Detector-usage-cn.md">简体中文</a> </center>

MetDetPy project provides two detection tools: `MetDetPy` for video meteor detection, and `MetDetPhoto` for image meteor detection. By combining with `ClipToolkit`, you can achieve the detection-export workflow for meteors.

## Table of Contents

* [Quick Start](#quick-start)
* [Tool Comparison](#tool-comparison)
* [MetDetPy - Video Meteor Detector](#metdetpy---video-meteor-detector)
    * [Complete Parameter Description](#metdetpy-complete-parameter-description)
    * [Usage Examples](#metdetpy-usage-examples)
* [MetDetPhoto - Image Meteor Detector](#metdetphoto---image-meteor-detector)
    * [Complete Parameter Description](#metdetphoto-complete-parameter-description)
    * [Usage Examples](#metdetphoto-usage-examples)
* [Output Description](#output-description)
* [FAQ](#faq)

---

## Quick Start

MetDetPy's detection tools are designed with sensible default configurations and can run directly without additional parameters in most cases.

### Tool Selection

- If it is a **directly recorded meteor video file** (fps>=4), use `MetDetPy`;
- If you have **static images**, an **image folder** (such as timelapse sequence images), or a **timelapse video**, use `MetDetPhoto`.

### Video Detection

```sh
# Detect video (results only output to command line)
python MetDetPy.py video.mp4

# Detect and save results
python MetDetPy.py video.mp4 --save-path results.json

# After detection, filter out all negative samples and small size results, export only all meteor segments as videos
python MetDetPy.py video.mp4 --save-path results.json
python ClipToolkit.py results.json --mode video --enable-filter-rules --save-path ./output

# Or export with bounding boxes
python ClipToolkit.py results.json --mode video --enable-filter-rules --with-bbox --save-path ./output
```

### Image Detection

```sh
# Detect image folder
python MetDetPhoto.py ./images

# Detect and save results
python MetDetPhoto.py ./images --save-path results.json

# After detection, filter out all negative samples and small size results, copy positive samples to another folder
python ClipToolkit.py results.json --enable-filter-rules  --save-path ./output

# Or export with bounding boxes
python ClipToolkit.py results.json --enable-filter-rules --with-bbox --save-path ./output
```

---

## Tool Comparison

| Feature | MetDetPy | MetDetPhoto |
|------|-----------|--------------|
| **Input Type** | Video files | Single image, image folder, timelapse video |
| **Detection Method** | Traditional algorithm + optional deep learning | Deep learning model |
| **Main Advantage** | Motion analysis, time series processing | Static image detection, fast |
| **Applicable Scenarios** | Meteor detection and motion analysis in videos | Meteor detection in static images, batch image processing |
| **Recheck Mechanism** | Supported (visual + motion properties) | Not applicable |

**Selection Guide:**
- If you have **video files**, use `MetDetPy`
- If you have **static images** or **image folders**, use `MetDetPhoto`
- If you need to leverage **time series information** and **motion analysis**, use `MetDetPy`
- If you need **fast batch processing** of static images, use `MetDetPhoto`

---

## MetDetPy - Video Meteor Detector

`MetDetPy` is the video meteor detector launcher of MetDetPy project, which can detect meteor events from video files.

### MetDetPy Complete Parameter Description

```sh
python MetDetPy.py target [--cfg CFG] [--mask MASK] [--start-time START_TIME] [--end-time END_TIME]
               [--exp-time EXP_TIME] [--mode {backend,frontend}] [--debug]
               [--resize RESIZE] [--adaptive-thre ADAPTIVE_THRE] [--bi-thre BI_THRE | --sensitivity SENSITIVITY]
               [--recheck RECHECK] [--save-rechecked-img SAVE_RECHECKED_IMG]
               [--provider {cpu,default,coreml,dml,cuda}] [--live-mode {on,off}] [--save-path SAVE-PATH]
               [--resource-dir RESOURCE_DIR]
```

#### Main Parameters

| Parameter | Description | Default Value |
|------|------|--------|
| `target` | Target video file path. Supports common video encodings such as H264, HEVC | Required |
| `--cfg` | Configuration file path | `./config/m3det_normal.json` |
| `--mask` | Mask (overlay) image path | None |
| `--start-time` | Detection start time (milliseconds or `"HH:MM:SS"` format) | 0 (video start) |
| `--end-time` | Detection end time (milliseconds or `"HH:MM:SS"` format) | Video end |
| `--mode` | Running mode: `frontend` (show progress bar) or `backend` (pipe mode) | `frontend` |
| `--debug` | Enable debug mode, print detailed logs | Off |
| `--visu` | Enable visualization window to show detection process in real-time | Off |
| `--live-mode` | Live mode: detection time close to actual video duration, balance CPU cost | `off` |
| `--provider` | Model inference backend: `cpu`, `default`, `coreml`, `dml`, `cuda` | `default` |
| `--save-path` | Save detection results to MDRF format JSON file path | Not saved |

#### Extra Parameters (Override Configuration File)

The following parameters use default values from configuration file when not set. For detailed information about configuration file, please refer to [Configuration File Documentation](../config-doc.md).

| Parameter | Description | Default Value |
|------|------|--------|
| `--resize` | Frame image size used during detection. Can be specified as integer (e.g., `960` for long side), list (e.g., `[960,540]`), or string (e.g., `960x540`) | Value in configuration file |
| `--exp-time` | Single frame exposure time. Can specify float or select from `{auto, real-time, slow}` | `auto` |
| `--adaptive-thre` | Whether to enable adaptive binary threshold. Select from `{on, off}` | Value in configuration file |
| `--bi-thre` | Binary threshold. This option is invalid when adaptive binary threshold is enabled. Cannot use with `--sensitivity` | Value in configuration file |
| `--sensitivity` | Detector sensitivity. Select from `{low, normal, high}`. Cannot use with `--bi-thre` | Value in configuration file |
| `--recheck` | Whether to enable recheck mechanism to reduce false positives. Select from `{on, off}` | Value in configuration file |
| `--save-rechecked-img` | Path to save rechecked images | Not saved |

### MetDetPy Usage Examples

A common combination parameter example is as follows (usually no need to enable all):

```sh
python MetDetPy.py video.mp4 \
    --mask mask.jpg \ # Use mask to exclude interference areas
    --save-path result.json \ # Save results to JSON file
    --start-time 00:01:00 --end-time 00:05:00 \ # Detect from 00:01:00 to 00:05:00
    --visu \ # Enable visualization window
    --debug \ # Enable detailed debug information
    --live-mode \ # Live mode
    --resize 960x540 \ # Frame image size used during detection
    --exp-time 1000 \ # Single frame exposure time
    --adaptive-thre on \ # Enable adaptive binary threshold
    --sensitivity normal \ # Detector sensitivity
```

---

## MetDetPhoto - Image Meteor Detector

`MetDetPhoto` is the image meteor detector launcher of MetDetPy project, which can detect meteors from single images, image folders, or timelapse videos.

### MetDetPhoto Complete Parameter Description

```sh
python MetDetPhoto.py target [--mask MASK]
                             [--model-path MODEL_PATH] [--model-type MODEL_TYPE]
                             [--exclude-noise] [--debayer] [--debayer-pattern DEBAYER_PATTERN]
                             [--visu] [--visu-resolution VISU_RESOLUTION]
                             [--save-path SAVE_PATH]
                             [--resource-dir RESOURCE_DIR]
```

#### Parameters

| Parameter | Description | Default Value |
|------|------|--------|
| `target` | Target image file/folder. Supports single image, image folder, and timelapse video files with common video encoding formats | Required |
| `--mask` | Mask (overlay) image path | None |
| `--model-path` | Model weight file path | `./weights/yolov5s_v2.onnx` |
| `--model-type` | Model format, determines how to process model outputs | `YOLO` |
| `--exclude-noise` | Exclude common noise categories (such as satellites and bugs), only save positive samples to files | Off |
| `--debayer` | Perform Debayer transform on video frames before processing timelapse video | Off |
| `--debayer-pattern` | Debayer matrix, such as `RGGB` or `BGGR`. Only works when `--debayer` option is enabled | None |
| `--visu` | Enable visualization window to show current detection situation | Off |
| `--visu-resolution` | Visualization window resolution setting | Default resolution |
| `--save-path` | Save detection results to [MDRF](../data-format.md#meteor-detection-recording-format-mdrf) format file | Not saved |

### MetDetPhoto Usage Examples

#### Basic Usage

```sh
# Detect a single image
python MetDetPhoto.py image.jpg

# Detect all images in an image folder
python MetDetPhoto.py ./images

# Detect meteor frames in timelapse video and save detection results
python MetDetPhoto.py timelapse.mp4  --save-path result.json
```

#### MetDetPhoto Usage Examples

A common combination parameter example is as follows (usually no need to enable all):

```sh
python MetDetPhoto.py ./images \
    --mask mask.jpg # Use mask to exclude interference areas \
    --model-path ./custom_model.onnx # Use custom model weight file \
    --exclude-noise # Only save meteor detection results, exclude satellites, bugs and other common noise \
    --debayer --debayer-pattern RGGB # Process RAW format timelapse video \
    --visu  --visu-resolution 1920x1080 # Enable visualization window \
    --save-path results.json # Save detection results to JSON file
```

---

## Output Description

### Command Line Output

Both detection tools will output detection information to command line in real-time during runtime, including:

- Detection progress
- Meteor sample information detected
- Runtime statistics

### Result File

When using `--save-path` parameter, detection results will be saved as a JSON file in [MDRF](../data-format.md#meteor-detection-recording-format-mdrf) format. This file contains:

- Video/image information (file path, resolution, frame rate, etc.)
- Detailed information of detection results
- Timestamp, bounding box coordinates, confidence score, etc. for each detection

This result file can be processed by tools like `ClipToolkit` to export meteor segments, generate stacked images, etc.

---

## FAQ

### Q1: How do I create a mask image?

**A:** Mask images are used to exclude areas that don't need detection (such as buildings, trees, etc.). Creation method:

1. Create a blank image (size不限，会自动缩放到视频/图像尺寸)
2. Use any non-white color to paint on areas that need to be excluded (recommended black or red)
3. Save as JPEG or PNG format

### Q2: What if detection results are not ideal?

**A:** You can try the following methods:

**For MetDetPy:**
- Adjust the configuration file used
- Use a mask image to exclude interference areas

**For MetDetPhoto:**
- Use a mask image to exclude interference areas
- Enable `--exclude-noise` to filter noise

### Q3: How do I export detected meteor segments?

**A:** Use `ClipToolkit` tool:

```sh
# 1. First run detection and save results
python MetDetPy.py video.mp4 --save-path results.json

# 2. Use ClipToolkit to export segments
python ClipToolkit.py results.json --mode video --save-path ./output
```

For details, see [ClipToolkit User Guide](./ClipToolkit-usage.md).

### Q4: What does `--live-mode` do?

**A:** `--live-mode` will control detection speed to be close to actual video duration, suitable for real-time monitoring scenarios, helping to balance CPU cost. Only applicable to `MetDetPy`.

### Q5: When do I need to use `--debayer`?

**A:** When processing RAW format timelapse videos (such as RAW videos captured with professional cameras), you need to use `--debayer` parameter for Debayer transformation. You need to specify the correct Bayer pattern according to the camera sensor settings (such as `RGGB`, `BGGR`, etc.).

---

## Related Documents

- [ClipToolkit User Guide](./ClipToolkit-usage.md)
- [Configuration File Documentation](../config-doc.md)
- [Data Format Documentation](../data-format.md)
- [Meteor Detection Recording Format (MDRF)](../data-format.md#meteor-detection-recording-format-mdrf)
