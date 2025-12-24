<div align="center">
  <img src="imgs/banner.png"/>

[![GitHub release](https://img.shields.io/github/release/LilacMeteorObservatory/MetDetPy.svg)](https://github.com/LilacMeteorObservatory/MetDetPy/releases/latest) [![GitHub Release Date](https://img.shields.io/github/release-date/LilacMeteorObservatory/MetDetPy.svg)](https://github.com/LilacMeteorObservatory/MetDetPy/releases/latest) [![license](https://img.shields.io/github/license/LilacMeteorObservatory/MetDetPy)](./LICENSE) [![Github All Releases](https://img.shields.io/github/downloads/LilacMeteorObservatory/MetDetPy/total.svg)](https://github.com/LilacMeteorObservatory/MetDetPy/releases)

<center>Language: English | <a href="./docs/readme-cn.md">简体中文</a></center>

</div>

## Introduction

MetDetPy is a Python-based meteor detector project that detects meteors from videos and images. Its video detection is inspired by [uzanka/MeteorDetector](https://github.com/uzanka/MeteorDetector). MetDetPy has the following features:

- **Easy-to-use and Configurable:** MetDetPy provides sensible default configurations so it works out-of-the-box in most situations, while also allowing configuration tweaks to improve detection results when needed.

- **Applicable for Various Devices and Exposure Times:** MetDetPy can detect meteors from videos and images captured by a wide range of devices. With adaptive algorithms and optional deep learning models, it works well for both meteor-monitoring cameras and conventional digital cameras.

- **Optional Deep Learning Integration:** Deep learning models can be optionally used in the main detection or recheck stage to improve results without significantly increasing runtime overhead. Models are also available for image-based meteor detection.

- **Effective Filters:** Detections are rechecked based on visual appearance and motion properties to reduce false positives. Each prediction is assigned a confidence score in [0,1], representing its likelihood to be a true meteor.

- **Support Tools:** MetDetPy ships several helper tools for evaluation and export, including an evaluation tool, a clip/stack toolkit, and packaging utilities.

## Release Version

You can get the latest release version of MetDetPy [here](https://github.com/LilacMeteorObservatory/MetDetPy/releases). The release artifacts are packaged for common platforms (Windows, macOS). You can also build standalone executables yourself using `nuitka` (see [Package python codes to executables](./docs/tool-usage.md#package-python-codes-to-executables)).

Besides, MetDetPy has worked as the backend of the Meteor Master since version 1.2.0. Meteor Master (AI) is a meteor detection software developed by [奔跑的龟斯](https://www.photohelper.cn), which has a well-established GUI, live streaming video support, convenient export function, automatic running, etc. You can get more information at [Meteor Master Official Site](https://www.photohelper.cn/MeteorMaster), or get its latest version from the Microsoft Store / App Store. Its earlier version can get from [Baidu NetDisk](https://pan.baidu.com/s/1B-O8h4DT89y_u1_YKXKGhA) (Access Code: jz01)

## Requirements

**Environment**

- 64bit OS
- Python>=3.7 (3.9+ is recommended)

**Python Requirements**

- numpy>=1.15.0
- opencv_python>=4.9.0
- tqdm>=4.0.0
- multiprocess>=0.70.0
- onnxruntime>=1.16.0
- av>=15.0.0
- dacite>=1.9.0
- pyexiv2>=2.12.0

You can install these requirements using:

```sh
pip install -r requirements.txt
```

### GPU Support

The above packages enable MetDetPy to run properly, but deep learning models in the default runtime are CPU-only on typical installations. If you wish to utilize your GPU, you can additionally install or replace the onnxruntime-related libraries as follows:

- **Windows / Linux (recommended):** install `onnxruntime-directml` to get DirectML-based acceleration on many GPUs (Nvidia, AMD, Intel). The package name on PyPI is `onnxruntime-directml`.

- **Nvidia GPU users (advanced):** if you have CUDA installed, install a CUDA-matched `onnxruntime-gpu` build instead of `onnxruntime` to enable CUDA acceleration.

#### ⚠️ Notice

- For macOS users, CoreML inference acceleration is already integrated into recent `onnxruntime` builds, so no extra step is normally required to enable GPU support on macOS.

- In the current packaged Windows release we use a DirectML-enabled runtime. Default CUDA wheels will be adopted when fully tested.

## Usage

### Run Video Meteor Detector

MetDetPy is the launcher of the video meteor detector, its usage is as follows:

```sh
python MetDetPy.py target [--cfg CFG] [--mask MASK] [--start-time START_TIME] [--end-time END_TIME]
               [--exp-time EXP_TIME] [--mode {backend,frontend}] [--debug] [--visual]
               [--resize RESIZE] [--adaptive-thre ADAPTIVE_THRE] [--bi-thre BI_THRE | --sensitivity SENSITIVITY]
               [--recheck RECHECK] [--save-rechecked-img SAVE_RECHECKED_IMG]
               [--provider {cpu,default,coreml,dml,cuda}] [--live-mode {on,off}] [--save-path SAVE-PATH]
```

#### Main Arguments

* `target`: meteor video filename. Support common video encoding like H264, HEVC, etc.

* `--cfg`: path to the configuration file. Use [./config/m3det_normal.json](./config/m3det_normal.json) under the config folder by default.

* `--mask`: mask image. To create a mask image, draw mask regions on a blank image using any color (except white). Support JPEG and PNG format.

* `--start-time`: the time at which the detection starts (an int in ms or a string format in `"HH:MM:SS"`). The default value is the start of the video (i.e., 0).

* `--end-time`: the time until which the detecting ends (an int in ms or a string format in `"HH:MM:SS"`). The default value is the end of the video.

* `--mode`: the running mode. Its argument should be selected from `{backend, frontend}`. In `frontend` mode, there will be a progress bar indicating related information. In `backend` mode, the progress information is flushed immediately to suit pipeline workflow. The default is `"frontend"`.

* `--debug`: indicates whether to print debug information.

* `--visual`: showing a debug window displaying videos and detected meteors.

* `--live-mode`: when running in live mode, the detection speed will closely match the actual video time. This option balance cpu cost. Should be selected from `{on, off}`.

* `--provider`: specifies the preferred provider to be used for models. The available providers may vary depending on the platform. If the specified provider is not available, the "default" option will be used.

* `--save-path`: save detection results to a json file in [MDRF](./docs/tool-usage.md#meteor-detection-recording-format-mdrf) format.

#### Extra Arguments

The following arguments have default values in config files. If they are configured in command line arguments, the default value will be overrided. Their detailed explanation can be seen in [configuration documents](./docs/config-doc.md).

* `--resize`: the frame image size used during the detection. This can be set by single int (like `960`, for the long side), list (like `[960,540]`) or string (like `960x540` or `1920x1080`).

* `--exp-time`: the exposure time of each frame in the video. Set with a float number or select from {auto, real-time, slow}. For most cases, option "auto" works well.

* `--adaptive-thre`: indicates whether apply adaptive binary threshold in the detector. Select from {on, off}.

* `--bi-thre`: the binary threshold used in the detector. When the adaptive binary threshold is applied, this option is invalidated. Do not set --sensitivity with this at the same time.

* `--sensitivity`: the sensitivity of the detector. Select from {low, normal, high}. When adaptive binary threshold is applied, higher sensitivity will estimate a higher threshold. Do not set --bi-thre with this at the same time.

* `--recheck`: indicates whether apply recheck mechanism. Select from {on, off}.

#### Example

```sh
python MetDetPy.py "./test/20220413Red.mp4" --mask "./test/mask-east.jpg" --visu --save-path .
```

#### Output

`MetDetPy` outputs the detection results to the command line for real-time verification during runtime. Specifying the `--save-path` parameter at runtime will also save the detection results to a specified file in `MDRF` format. The detection result file can be processed by the `MetDetPy` project's [Other Tools](#Usage-of-Other-Tools) to further generate content such as meteor fragments, meteor screenshots, and annotation files.

For instructions on using these tools, please refer to the [Tool Documentation](./docs/tool-usage.md); for information on the meaning of the output fields, please refer to the [Data Format](./docs/data-format.md).

#### Customize Configuration

MetDetPy reads arguments from configuration files. For most circumstances, preset configuration files work well, but there are also times when better detection results can be achieved by adjusting detection arguments. This document explains the meanings of arguments so that they can adjusted according to the requirement. See [configuration documents](./docs/config-doc.md) for more information.

### Run Image Meteor Detector

`MetDetPhoto` is the launcher of the image meteor detector, its usage is as follows:

```sh
python MetDetPhoto.py target [--mask MASK]
                             [--model-path MODEL_PATH] [--model-type MODEL_TYPE]
                             [--exclude-noise] [--debayer] [--debayer-pattern DEBAYER_PATTERN]
                             [--visu] [--visu-resolution VISU_RESOLUTION]
                             [--save-path SAVE_PATH]
```

#### Arguments

* `target`: meteor image target, support single image, image folder and a timelapse video with common video encoding.

* `--mask`: mask image. To create a mask image, draw mask regions on a blank image using any color (except white). Support JPEG and PNG format.

* `--model-path`: path to the model weight file. Use [./weights/yolov5s_v2.onnx](./weights/yolov5s_v2.onnx) as the default weight file.

* `--model-type`: the type of the model. For now only `YOLO` is supported. Default to `YOLO`.

* `--exclude-noise`: exclude common noise category (like satellites and bugs) from predictions, only save positive samples to files.

* `--debayer`: whether to execute debayer transform for timelapse video before detection.

* `--debayer-pattern`: debayer pattern, like RGGB or BGGR. Only work when `--debayer` is applied.

* `--visu`: showing a debug window displaying images and detected meteors.

* `--visu-resolution`: visualized debug window resolution.

* `--save-path`: save detection results to a json file in [MDRF](./docs/tool-usage.md#meteor-detection-recording-format-mdrf) format.

#### Example

```sh
python MetDetPhoto.py "/path/to/your/folder" --mask "/path/to/your/mask.jpg" --exclude-noise --save-path .
```

### Usage of Other Tools

Several tools are provided with MetDetPy to support related functions, including ClipToolkit (batch image stacking and video clipping tool), Evaluate (performance evaluation and regression testing tool), and make_package (packaging script). Access the [tool documentation](./docs/tool-usage.md) to learn more about how to use these tools.

## Performance and Efficiency

1. When applying default configuration on 3840x2160 10fps video, MetDetPy detect meteors with a 20-30% time cost of video length on average (tested with an Intel i5-7500). Videos with higher FPS may cost more time.

2. We test MetDetPy with videos captured from various devices (from modified monitoring cameras to digital cameras), and MetDetPy achieves over 80% precision and over 80% recall on average.

3. MetDetPy is fast and efficient at detecting most meteor videos. However, when facing complicated weather or other affecting factors, its precision and recall can be improved. If you find that MetDetPy does not perform well enough on your videos, you are welcome to contact us or submit an issue (please attach full or clipped videos when applicable).

## License

This project is licensed under the Mozilla Public License 2.0 (MPL-2.0). This means you are free to use, modify, and distribute this software with the following conditions:

1. **Source Code Availability**: Any modifications you make to the source code must also be made available under the MPL-2.0 license. This ensures that the community can benefit from improvements and changes.
2. **File-Level Copyleft**: You can combine this software with other code under different licenses, but any modifications to the MPL-2.0 licensed files must remain under the same license.
3. **No Warranty**: The software is provided "as-is" without any warranty of any kind, either express or implied. Use it at your own risk.

For more detailed information, please refer to the [MPL-2.0 license text](https://www.mozilla.org/en-US/MPL/2.0/).

## Appendix

### Special Thanks

uzanka [[Github]](https://github.com/uzanka)

奔跑的龟斯 [[Personal Website]](https://photohelper.cn) [[Weibo]](https://weibo.com/u/1184392917) [[Bilibili]](https://space.bilibili.com/401484)

纸片儿 [[Github]](https://github.com/ArtisticZhao)

DustYe夜尘 [[Bilibili]](https://space.bilibili.com/343640654)

RoyalK [[Weibo]](https://weibo.com/u/2244860993) [[Bilibili]](https://space.bilibili.com/259900185)

MG_Raiden扬 [[Weibo]](https://weibo.com/811151123) [[Bilibili]](https://space.bilibili.com/11282636)

星北之羽 [[Bilibili]](https://space.bilibili.com/366525868/)

LittleQ

韩雅南

来自偶然

杨雳鹏

兔爷 [[Weibo]](https://weibo.com/u/2094322147)[[Bilibili]](https://space.bilibili.com/1044435613)

Jeff戴建峰 [[Weibo]](https://weibo.com/1957056403) [[Bilibili]](https://space.bilibili.com/474329765)

贾昊

### Update Log / Todo List

See [update log](docs/update-log.md).

### Statistics

[![Star History Chart](https://api.star-history.com/svg?repos=LilacMeteorObservatory/MetDetPy&type=Timeline)](https://star-history.com/#LilacMeteorObservatory/MetDetPy&Timeline)
