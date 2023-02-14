# MetDetPy

Other Language Version: [[中文版]](./docs/readme-cn.md)

MetDetPy is a python-based video meteor detector that can detect meteors from video files.

* Basically, MetDetPy is inspired by [uzanka/MeteorDetector](https://github.com/uzanka/MeteorDetector). In this project, their work is also reproduced in python3.

* Based on their work, we further implement the M3 detector. The M3 detector works fine for videos with exposure time from 1/120s to 1/4s. It calculates the difference frame (calculated by maximum minus mean) in a wider sliding time window efficiently to improve accuracy.

* We design an adaptive threshold algorithm that can select binary threshold dynamically according to the signal-to-noise ratio of the video. (Experimental feature)

* We also implement a meteor detection result manager (called MeteorLib) to help integrate predictions and exclude false positive samples. Every prediction is given a confidence score ranging [0,1] which indicates the possibility of being considered a meteor.

* An evaluation tool is under development and coming soon.

## Release Version

You can get the latest release version of MetDetPy [here](https://github.com/LilacMeteorObservatory/MetDetPy/releases). The release version are already packed and can run on common platforms (including Windows, macOS and Linux) respectively. Also, you can build it yourself with `pyinstaller` or `nuitka` (see [Package python codes to executables](#package-python-codes-to-executables)).

Besides, MetDetPy works as the backend of the Meteor Master since version 1.2.0. Meteor Master is a video meteor detection software developed by [奔跑的龟斯](https://www.photohelper.cn), which has a well-established GUI, live streaming video support, convenient export function,  automatic running, etc. You can get the latest Meteor Master (Windows and macOS release version) from:

* [Meteor Master Official Site](https://www.photohelper.cn/MeteorMaster)
* [Baidu NetDisk](https://pan.baidu.com/s/1B-O8h4DT89y_u1_YKXKGhA) (Access Code: jz01)

## Requirements

### Enviroments

* 64bit OS
* Python>=3.7

### Packages

* numpy>=1.15.0
* opencv_python>=4.7.0
* tqdm>=4.0.0

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

* --start-time: the start time of detecting (in ms). The default is the start of the video (i.e, 0).

* --end-time: the end time of detecting (in ms). The default is the end of the video.

* --mode: the running mode. Its argument should be selected from {backend, frontend}. In frontend mode, there will be a progress bar indicating related information. In backend mode, the progress information is flushed immediately to suit pipeline workflow.  The default is "frontend".

* --debug: when launching MetDetPy with --debug, there will be a debug window showing videos and detected meteors.

#### Cover arguments

The following arguments has default value in [config files](./config.json). Their detail explanation can be seen in [configuration documents](./docs/config-doc.md).

* --resize: the frame image size used during the detection.

* --exp-time: the exposure time of each frame in the video. Set with a float number or select from {auto, real-time, slow}. For most cases, option "auto" works well.

* --adaptive-thre: whether apply adaptive binary threshold in the detector. Select from {on, off}.

* --bi-thre: the binary threshold used in the detector. When adaptive binary threshold is applied, this option is invalidate. Do not set --sensitivity with this at the same time.

* --sensitivity: the sensitivity of the detector. Select from {low, normal, high}. When adaptive binary threshold is applied, higher sensitivity will estimate a higher threshold. Do not set --bi-thre with this at the same time.

#### Example

```sh
python core.py ./test/20220413Red.mp4 --mask ./test/mask-east.jpg
```

### Customize Configuration

Unlike video-related arguments, most detect-related important arguments are predefined and stored in the configuration file. In most cases, predefined arguments works fine. However, sometimes it is possible to finetune these arguments to get better results. If you want to get the illustration of the configuration file, see [configuration documents](./docs/config-doc.md) for more information.

### Run Evaulation (Coming soon)

To evaluate this program on a series of videos, you can simply run `evaluate.py` :

```sh
python evaluate.py --videos test_video.json
```

where `test_video.json` places a series of videos and masks (if provided). It should be formatted like:

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

If a video has no corresponding mask, simply use `""` .

### Usage of Other Tools

#### ClipToolkit

ClipToolkit can be used to create several video clips or stacked images at once. Its usage is as follows:

```sh
python ClipToolkit.py [--mode {image,video}] [--suffix SUFFIX] [--save-path SAVE_PATH] [--resize RESIZE] [--jpg-quality JPG_QUALITY] [--png-compressing PNG_COMPRESSING] target json
```

positional arguments:

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
python ./ClipToolkit.py ./test/20220413Red.mp4 ./test/clip_test.json --mode image --suffix jpg --jpg-quality 60 --resize 960x540
```

Notice: if using a JSON-format string instead of the path to a JSON file, you should be really careful about the escape of double quotes in command lines.

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

Notice:

1. It is suggested to use `Python>=3.9`, `pyinstaller>=5.0`, and `nuitka>=1.3.0` to avoid compatibility issues.
2. According to our test, `pyinstaller` packages MetDetPy faster, and generated executables are usually smaller (about 30% smaller than its nuitka version). However, its executables may spend more time when launching. In contrast, `nuitka` takes more time at compiling and generates bigger executables (even with UPX compressing), but it launches faster (over 50%). Except for the launch time, their running time is mostly the same. Thus, you can choose the proper packaging tool to fit your requirement.
3. Due to the feature of Python, neither tools above can generate cross-platform executable files.

## Todo List

 1. 改善对于实际低帧率视频的检测效果 (Almost Done, but some potential bugs left)
    1. 找到合适的超参数： max_gap
    2. 设计再校验机制：利用叠图结果做重校准
    3. 优化速度计算逻辑，包括方向，平均速度等
    4. 改善自适应阈值：当误检测点很多时，适当提高分割阈值
 2. 改善对蝙蝠/云等情况的误检(!!)
 3. 支持导出UFO Analizer格式的文本，用于流星组网联测等需求
 4. 评估系统
 5. 利用cython改善性能
 6. 添加天区解析功能，为支持快速叠图，组网提供基础支持
 7. 改善解析帧率与真实帧率不匹配时的大量误报问题；优化帧率估算机制；优化关键帧附近大量噪点的问题

P.S: 目前结合MeteorMaster已支持/将支持以下功能，它们在MetDetPy的开发中优先级已下调：

 1. 完善的GUI
 2. 支持rtmp/rtsp/http流直播
 3. 时间水印（待开发）
 4. 自动启停

## Performance and Efficiency

 1. With `MergeStacker`, MetDetPy now can detect meteors with a 20-30% time cost of video length on average (tested with an Intel i5-7500).

 2. Test tool `evaluate.py` is going to be updated soon. For now, MetDetPy performs great for videos from monitoring cameras. For camera-captured videos, the ratio of false positive samples still seems to be a little high.

## Appendix

### Special Thanks

uzanka [[Github]](https://github.com/uzanka)

奔跑的龟斯 [[Personal Website]](https://photohelper.cn) [[Weibo]](https://weibo.com/u/1184392917)

纸片儿 [[Github]](https://github.com/ArtisticZhao)

DustYe夜尘[[Bilibili]](https://space.bilibili.com/343640654)

RoyalK[[Weibo]](https://weibo.com/u/2244860993) [[Bilibili]](https://space.bilibili.com/259900185)

MG_Raiden扬[[Weibo]](https://weibo.com/811151123) [[Bilibili]](https://space.bilibili.com/11282636)

星北之羽[[Bilibili]](https://space.bilibili.com/366525868/)

LittleQ

韩雅南

来自偶然

ylp

### Update Log

See [update-log](docs/update-log.md).
