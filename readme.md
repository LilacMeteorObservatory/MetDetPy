<div align="center">
  <img src="imgs/banner.png"/>

![version](https://img.shields.io/badge/version-v2.0.0_alpha1-success) [![license](https://img.shields.io/badge/license-MPL2.0-success)](./LICENSE) 

<center>Language: English | <a href="./docs/readme-cn.md">简体中文</a></center>

</div>

## Introduction

MetDetPy is a python-based video meteor detector that can detect meteors from video files, which is inspired by [uzanka/MeteorDetector](https://github.com/uzanka/MeteorDetector). MetDetPy is more powerful and reliable, with the following features:

* **Adaptive Sensitivity:** for most meteor videos, MetDetPy can be used directly without detailed configuration. With a series of adaptive algorithms, MetDetPy can change its detecting sensitivity according to the signal-to-noise ratio of the video.

* **Applicable for Various Devices and Exposure Time:** MetDetPy can detect meteors from video files that capture by various types of devices. We implement the M3 detector, which works fine for videos with exposure time from 1/120s to 1/4s. It calculates the difference frame (calculated by maximum minus mean) in a wider sliding time window efficiently to improve accuracy.

* **Low CPU and Memory Usage:** MetDetPy is developed based on OpenCV, thus it works with low CPU and memory usage while not requiring GPU. It can support multi-camera real-time detection on mainstream computers or barebones.

* **Effective Filter:** a meteor detection result manager (called MeteorLib) is designed to help integrate predictions and exclude false positive samples. Every prediction is given a confidence score ranging [0,1] which indicates the possibility of being considered a meteor.

* **Abundant Support Tools:** an evaluation tool and a video clip toolkit are also provided to support further video clipping, image stacking, or result evaluation.

## Release Version

You can get the latest release version of MetDetPy [here](https://github.com/LilacMeteorObservatory/MetDetPy/releases). The release version are already packed and can run on common platforms (including Windows, macOS and Linux) respectively. Also, you can build it yourself with `pyinstaller` or `nuitka` (see [Package python codes to executables](#package-python-codes-to-executables)).

Besides, MetDetPy works as the backend of the Meteor Master since version 1.2.0. Meteor Master is a video meteor detection software developed by [奔跑的龟斯](https://www.photohelper.cn), which has a well-established GUI, live streaming video support, convenient export function,  automatic running, etc. You can get the latest Meteor Master (Windows and macOS release version) from:

* [Meteor Master Official Site](https://www.photohelper.cn/MeteorMaster)
* [Baidu NetDisk](https://pan.baidu.com/s/1B-O8h4DT89y_u1_YKXKGhA) (Access Code: jz01)

## Requirements

### Enviroments

* 64bit OS
* Python>=3.7 (3.9+ is recommended)

### Packages

* numpy>=1.15.0
* opencv_python>=4.7.0
* tqdm>=4.0.0
* easydict>=1.0
* multiprocess>=0.70.0
* onnxruntime>=1.16.0

You can install these packages using:

```sh
pip install -r requirements.txt
```

## Usage

### Run MetDetPy

```sh
python core.py target [--cfg CFG] [--mask MASK] [--start-time START_TIME] [--end-time END_TIME] 
               [--exp-time EXP_TIME] [--mode {backend,frontend}] [--debug]
               [--resize RESIZE] [--adaptive-thre ADAPTIVE_THRE] [--bi-thre BI_THRE | --sensitivity SENSITIVITY]
```

#### Main Arguments

* target: meteor video filename. Support common video encoding (since it uses OpenCV to decode video).

* --cfg: configuration file. Use [config.json](./config.json) under the same path default.

* --mask: mask image. To create a mask image, draw mask regions on a blank image using any color (except white). Support JPEG and PNG format.

* --start-time: the time at which the detection starts (an int in ms or a string format in `"HH:MM:SS"`). The default value is the start of the video (i.e, 0).

* --end-time: the time until which the detecting ends (an int in ms or a string format in `"HH:MM:SS"`). The default value is the end of the video.

* --mode: the running mode. Its argument should be selected from {backend, frontend}. In frontend mode, there will be a progress bar indicating related information. In backend mode, the progress information is flushed immediately to suit pipeline workflow.  The default is "frontend".

* --debug: when launching MetDetPy with --debug, there will be a debug window showing videos and detected meteors.

#### Cover arguments

The following arguments has default value in [config files](./config.json). Their detail explanation can be seen in [configuration documents](./docs/config-doc.md).

* --resize: the frame image size used during the detection. This can be set by single int (like `960`, for the long side), list (like `[960,540]`) or string (like `960x540` or `1920x1080`).

* --exp-time: the exposure time of each frame in the video. Set with a float number or select from {auto, real-time, slow}. For most cases, option "auto" works well.

* --adaptive-thre: whether apply adaptive binary threshold in the detector. Select from {on, off}.

* --bi-thre: the binary threshold used in the detector. When adaptive binary threshold is applied, this option is invalidate. Do not set --sensitivity with this at the same time.

* --sensitivity: the sensitivity of the detector. Select from {low, normal, high}. When adaptive binary threshold is applied, higher sensitivity will estimate a higher threshold. Do not set --bi-thre with this at the same time.

#### Example

```sh
python core.py "./test/20220413Red.mp4" --mask "./test/mask-east.jpg"
```

### Customize Configuration

Unlike video-related arguments, most detect-related important arguments are predefined and stored in the configuration file. In most cases, predefined arguments works fine. However, sometimes it is possible to finetune these arguments to get better results. If you want to get the illustration of the configuration file, see [configuration documents](./docs/config-doc.md) for more information.

### Usage of Other Tools

#### ClipToolkit

ClipToolkit can be used to create several video clips or stacked images at once. Its usage is as follows:

```sh
python ClipToolkit.py [--mode {image,video}] [--suffix SUFFIX] [--save-path SAVE_PATH] [--resize RESIZE] [--jpg-quality JPG_QUALITY] [--png-compressing PNG_COMPRESSING] target json
```

##### Arguments:

* target: the target video.

* json: a JSON-format string or the path to a JSON file where start time, end-time and filename (optional) are listed.

    Specifically, this JSON should be an array of elements, where every element should include at least a `"time"` key. The value of the `"time"` key should be an array of two `"hh:mm:ss.ms"` format strings, which indicates the start time and the end time of the clip. `"filename"` is an optional key, in whose value you can specify the filename and suffix (i.e what the video clip should be converted to and named.) `"filename"` is more prior than `--mode` and `--suffix` options, but if not specified, this clip will be automatically converted and named according to the command options.

    We provide [clip_test.json](./test/clip_test.json) as a use case and test JSON.

* --mode: convert clip(s) to images or videos. Should be selected from {image, video}. This option will be covered by a specific filename in json.

* --suffix: the suffix of the output. By default, it is "jpg" for image mode and "avi" for video mode. This option will be covered by a specific filename in JSON.

* --save-path: the path where image(s)/video(s) are placed. When only one clip is provided in JSON, you can include the filename in --save-path to simplify your JSON.

* --resize: resize image/video to the given resolution. It should be a string where two numbers are joined by `x`,like `960x540` or `1920x1080`.

* --png-compressing: the compressing rate of the generated png image. It should be int ranged $Z \in [0,9]$; By default, it is 3.

* --jpg-quality: the quality of generated jpg image. It should be int ranged $Z \in [0,100]$; By default, it is 95.

For example:

```sh
python ClipToolkit.py "./test/20220413Red.mp4" "./test/clip_test.json" --mode image --suffix jpg --jpg-quality 60 --resize 960x540
```

Notice: if using a JSON-format string instead of the path to a JSON file, you should be really careful about the escape of double quotes in command lines.


#### Evaulate

To evaluate how MetDetPy performs on your video, you can simply run `evaluate.py` :

```sh
python evaluate.py target [--cfg CFG] [--load LOAD] [--save SAVE] [--metrics] [--debug] video_json
```
##### Arguments

* video_json: a JSON file that places the name of the video, the mask, and meteor annotations. It should be formatted like this:

```json
{
    "video": "path/to/the/video.mp4",
    "mask": "path/to/the/mask.jpg",
    "meteors": [{
        "start_time": "HH:MM:SS.XX0000",
        "end_time": "HH:MM:SS.XX0000",
        "pt1": [
            260,
            225
        ],
        "pt2": [
            154,
            242
        ]
    }]
}
```

If there is no corresponding mask, simply use `""`. If there is no meteor annotation, the `"meteors"` can be ignored too.

* --cfg: configuration file. Use [config.json](./config.json) under the same path default.

* --load: the filename of the detection result that is saved by `evaluate.py`. If it is applied, `evaluate.py` will directly load the result file instead of running detection through the video.

* --save: the filename of the detection result that is going to save.

* --metrics: calculate precision and recall of the detection. To apply this, `"meteors"` has to be provided in `video_json`.

* --debug: when launching `evaluate.py` with this, there will be a debug window showing videos and detected meteors.

##### Example
```sh
python evaluate.py "test/20220413_annotation.json"
```


## Package python codes to executables

We provide [make_package.py](make_package.py) to freeze MetDetPy programs into stand-alone executables. This tool supports to use `pyinstaller` or `nuitka` to package/compile MetDetPy (and related tools).

When using it, make sure that either `pyinstaller` or `nuitka` is installed. Besides, when using `nuitka` as the packaging tool, make sure that at least one C/C++ compiler is available on your computer.

Its usage is as follows:

```sh
python make_package.py [--tool {nuitka,pyinstaller}] [--mingw64]
     [--apply-upx] [--apply-zip] [--version VERSION]
```

* --tool: your compile/package tool. It should be selected from {nuitka,pyinstaller}. `nuitka` is the default option.

* --mingw64: use the mingw64 compiler. Only worked when using `nuitka` and your OS is windows.

* --apply-upx: apply UPX to squeeze the size of the executable program. Only worked when using `nuitka`.

* --apply-zip: generate zip package when compiling/packaging is finished.

* --version: MetDetPy version tag. Used for naming zip package.

The target executable file and its zip package version (if applied) will be generated in  [dist](./dist/)  directory.

**Notice:**

1. It is suggested to use `Python>=3.9`, `pyinstaller>=5.0`, and `nuitka>=1.3.0` to avoid compatibility issues. Besides, avoid using `nuitka>=1.5.0` (2023.03), which might lead to SystemError on some devices.
2. According to our test, `pyinstaller` packages MetDetPy faster, and generated executables are usually smaller (about 30% smaller than its nuitka version). However, its executables may spend more time when launching. In contrast, `nuitka` takes more time at compiling and generates bigger executables (even with UPX compressing), but it launches faster (over 50%). Except for the launch time, their running time is mostly the same. Thus, you can choose the proper packaging tool to fit your requirement.
3. Due to the feature of Python, neither tools above can generate cross-platform executable files.
4. If `matplotlib` or `scipy` is in the environment, they are likely to be packaged into the final directory together. To avoid this, it is suggested to use a clean environment for packaging.

## Todo List

 1. 改善检测效果 (Almost Done, but some potential bugs left)
    1. 设计再校验机制：利用叠图结果做重校准
    2. 优化速度计算逻辑，包括方向，平均速度等
    3. 改善对暗弱流星的召回率
    4. 改善解析帧率与真实帧率不匹配时的大量误报问题
    5. 优化帧率估算机制；
    6. 改善对蝙蝠/云等情况的误检(？)
 2. 支持导出UFO Analizer格式的文本，用于流星组网联测等需求
 3. 利用cython改善性能
 4. 添加天区解析功能，为支持快速叠图，分析辐射点，流星组网监测提供基础支持
 

P.S: 目前结合MeteorMaster已支持/将支持以下功能，它们在MetDetPy的开发中优先级已下调：

 1. 完善的GUI
 2. 支持rtmp/rtsp/http流直播
 3. 时间水印（待开发）
 4. 自动启停

## Performance and Efficiency

1. When applying default configuration on 3840x2160 10fps video, MetDetPy detect meteors with a 20-30% time cost of video length on average (tested with an Intel i5-7500). Videos with higher FPS may cost more time.

2. So far no deep-learning model is introduced to MetDetPy, thus it does not require GPU and can support multi-camera real-time detection on mainstream computers or barebones ([MeteorMaster](https://www.photohelper.cn/MeteorMaster) has supported this). (P. S: We do plan to add A simple and lightweight CNN classifier in our future ---- do not worry, it will not increase CPU load significantly, while it can utilize Nvidia GPU if applicable.)

3. We test MetDetPy with videos captured from various devices (from modified monitoring cameras to digital cameras), and MetDetPy achieves over 80% precision and over 80% recall on average.

4. MetDetPy now is fast and efficient at detecting most meteor videos. However, when facing complicated weather or other affect factors, its precision and recall can be to be improved. If you find that MetDetPy performs not well enough on your video, it is welcome to contact us or submit issues (if possible and applicable, provide full or clipped video).

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

### Update Log

See [update log](docs/update-log.md).
