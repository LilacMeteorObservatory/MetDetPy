# MetDetPy

Other Language Version: [[中文版]](./docs/readme-cn.md)

MetDetPy is a python-based video meteor detector that can detect meteors from video files.

* Basically, MetDetPy is enlightened by [uzanka/MeteorDetector](https://github.com/uzanka/MeteorDetector). In this project, their work is reproduced in python3.

* Based on their work, we implement an M3 detector. The M3 detector works fine for videos with exposure time from 1/120s to 1/4s. It calculates the difference frame (calculated by maximum minus mean) in a wider sliding time window efficiently to improve accuracy.

* We design an adaptive threshold algorithm that can select binary threshold dynamically according to the signal-to-noise ratio of the video. (Experimental feature)

* We also implement a meteor detection result manager (called MeteorLib) to help integrate predictions and exclude false positive samples. Every prediction is given a confidence score ranging [0,1] which indicates the possibility of being considered a meteor.

* An evaluation tool is under development.

## Release Version

We provide release versions of MetDetPy for common platforms (including Windows, macOS and Linux). Also, you can build it yourself with `pyinstaller` or `nuitka` (see [Package python codes to executables](#package-python-codes-to-executables)).

Besides, MetDetPy works as the backend of the Meteor Master since version 1.2.0. You can get MeteorMaster(Windows release) version from:

* [Photohelper.cn](https://www.photohelper.cn/MeteorMaster)
* [Baidu NetDisk](https://pan.baidu.com/s/1B-O8h4DT89y_u1_YKXKGhA) (Access Code: jz01)



## Requirements

### Enviroments

* Python>=3.6

### Packages

* numpy>=1.15.0
* opencv_python>=4.7.0
* tqdm>=4.0.0

You can install these packages using:

```sh
pip install -r requirements.txt
```

## Usage

### Run Directly

```sh
python core.py target [--cfg CFG] [--mask MASK] [--start-time START_TIME] [--end-time END_TIME] 
               [--exp-time EXP_TIME] [--mode {backend,frontend}] [--debug]
               [--resize RESIZE] [--adaptive-thre ADAPTIVE_THRE] [--bi-thre BI_THRE | --sensitivity SENSITIVITY]
```

#### Main Arguments

* target: meteor video filename. Support common video encoding (since it uses OpenCV to decode video).

* --cfg: configuration file. Use "config.json" under the same path default.

* --mask: mask image. To create a mask image, draw mask regions on a blank image using any color (except white). Support JPEG and PNG format.

* --start-time: the start time of detecting (in ms). The default is 0.

* --end-time: the end time of detecting (in ms). The default is the end of the video.

* --mode: the running mode. Its argument should be selected from {backend, frontend}. In frontend mode, there will be a progress bar indicating related information. In backend mode, the progress information is flushed immediately to suit pipeline workflow.  The default is "frontend".

* --debug: when launching MetDetPy with --debug, there will be a debug window showing videos and detected meteors.

#### Cover arguments

The following arguments has default value in config files. Their detail explanation can be seen in [configuration documents](./docs/config-doc.md).

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

### Evaulate

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

## Package python codes to executables

We provide [make_package.py](make_package.py) to freeze MetDetPy programs into stand-alone executables. When using it, make sure that `pyinstaller` or `nuitka` is installed through `pip`. Also, when using `nuitka` as the packaging tool, make sure that at least one C/C++ compiler is available on your computer.

When everything is ready, run `python make_package.py` to package the code. By default `pyinstaller` is used as the preferred packaging tool, and you can add `--tool nuitka` to using `nuitka` to compile the code. Also, when using `nuitka` and your compiler is mingw64, add `--mingw64` to use the mingw64 compiler.

The target executable file and its zip package version will be generated in  [dist](./dist/)  directory.

Notice:
1. It is suggested to use `Python>=3.7` to avoid compatibility issues.
2. Due to the feature of Python, neither tools above can generate cross-platform executable files.

## Todo List

 1. 改善对于实际低帧率视频的检测效果 (Almost Done, but some potential bugs left)
    1. 找到合适的超参数： max_gap
    2. 设计再校验机制：利用叠图结果做重校准
    3. 优化速度计算逻辑，包括方向，平均速度等
    4. 改善自适应阈值：当误检测点很多时，适当提高分割阈值
 2. 改善对蝙蝠/云等情况的误检(!!)
 3. 完善日志系统
 4. 支持导出UFO Analizer格式的文本，用于流星组网联测等需求
 5. 快速叠图
 6. 评估系统
 7. 利用cython改善性能



P.S: 目前结合MeteorMaster已支持/将支持以下功能，它们在MetDetPy的开发中优先级已下调：

 1. 完善的GUI
 2. 支持rtmp/rtsp/http流直播
 3. 时间水印（待开发）
 4. 自动启停（待开发）

## Performance and Efficiency

 1. With `MergeStacker`, MetDetPy now can detect meteors with a 20-30% time cost of video length on average (tested with an Intel i5-7500).

 2. Test tool `evaluate.py` is going to be updated soon. For now, MetDetPy performs great for videos from monitoring cameras. For camera-captured videos, the ratio of false positive samples still seems to be a little high.

## Appendix

### Special Thanks

[uzanka](https://github.com/uzanka)

[奔跑的龟斯](https://weibo.com/u/1184392917)

[纸片儿](https://github.com/ArtisticZhao)

[DustYe夜尘](https://space.bilibili.com/343640654)

[RoyalK](https://weibo.com/u/2244860993)

[MG_Raiden扬](https://weibo.com/811151123)

[星北之羽](https://space.bilibili.com/366525868/)

LittleQ

### Update Log

#### Version 1.2.1

✅ Bug Fixed:
    Fix the exception output of the start time under some conditions / 修复了部分情况下起始时间异常的问题；
    Fix the exception abort issue during the FPS estimation / 修复了帧率估算时异常结束的问题；
    Fix the infinite loop issue during the FPS estimation / 修复了帧率估算时Sigma裁剪均值无法收敛的问题.

✅ Update packaging script: add Nuitka option, update spec file of pyinstaller. / 更新打包脚本：增加Nuitka打包选项，更新pyinstaller使用的.spec文件。

✅ Formally release packaging version for linux and macOS platforms. / 正式发布macOS和linux平台的打包版本。

#### Version 1.2

✅ Resolution-related APIs have been improved to support videos with different aspect ratios / 优化了分辨率相关接口以支持不同长宽比的视频

✅ Implemented an adaptive binary threshold mechanism. We explored simple empirical relations between std and the binary threshold. (⚠️This may not be appropriate for all video patterns. If you encounter any issues, please feel free to modify the function or help me improve it:) / 实现根据方差计算的自适应二值化阈值

✅ Sensitivity configuration is supported now: you can modify "sensitivity" under "bi_cfg" in config.json or passing arguments "--sensitivity {option}" to the program. / 增加灵敏度设置

✅ Smoothing probabilities to [0,1] instead of {0,1}. Meteor judging now is by a series of factors so it could be more accurate (perhaps...). / 输出调整为概率形式

#### Version 1.1.1

✅ Improved non-ASCII filename support for packaging version /改善对非ASCII文字路径的支持: by packaging with Python 3.7 and later, the new `pyinstaller` (>=5) can support non-ASCII paths and filenames.

#### Version 1.1

✅ Add "Backend" mode to support MeteorMaster. / 增加了后端模式

✅ Update output stream format. / 调整输出流格式

#### Version 1.0

Initial Version.