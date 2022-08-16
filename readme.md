# MetDetPy

Other Language Version: [[中文版]](./docs/readme-cn.md)

MetDetPy is a python script project that can detect meteors from video files. MetDetPy is enlightened by [MeteorDetector](https://github.com/uzanka/MeteorDetector).

* We reproduce their work and implement more detectors (like M3Detector (where M3 for maximum minus median)). These detectors can help realize highly sensitive meteor detection.

* We also implement a meteor detection result manager (MeteorLib) to help filter and integrate detection response, which is helpful to exclude false positive samples.

* An evaluation code is under development.

## Release Version

MetDetPy works as the backend of the Meteor Master since version 1.2.0. You can get MeteorMaster(Windows release) version from:

* [Photohelper.cn](https://www.photohelper.cn/MeteorMaster)
* [Baidu NetDisk](https://pan.baidu.com/s/1B-O8h4DT89y_u1_YKXKGhA) (Access Code: jz01)

We are planning to provide release versions of MetDetPy for other platforms (like macOS and Linux). Before which you can build it with `pyinstaller` (see [Package python codes to executables](#package-python-codes-to-executables)).

## Requirements

### Enviroments

* Python>=3.6

### Packages

* numpy>=1.15.0
* opencv_python>=4.0.0
* tqdm>=4.0.0

You can install these packages using:

```sh
pip install -r requirements.txt
```

## Usage

### Run Directly

```sh
python core.py  [--cfg CFG] [--mask MASK] 
                [--mode {backend,frontend}] 
                [--start-time START_TIME] [--end-time END_TIME]
                [--debug-mode] 
                target
```

#### Configs

* target: 待检测视频。目前主要支持H264编码视频。
* --cfg: 指定配置文件。默认情况下使用同目录下的config.json文件。
* --mask：指定掩模（遮罩）图像。使用黑色（任何颜色）涂抹不需要检测的区域即可。不强制要求尺寸与原图相同。默认不使用。
* --mode：指定以前端方式运行（即命令行直接启动）或作为后端被调用。两种情况下的输出流会存在一定差异。默认为前端模式。
* --start-time：指定从选定的时间开始分析。单位为ms。不指定将从头开始分析。
* --end-time：指定分析到从选定的时间。单位为ms。不指定将分析到视频结尾。

#### Example

```sh
python core.py ./test/20220413Red.mp4 --mask ./test/mask-east.jpg
```

#### Customize Configuration

Series of arguments can be defined and set in command lines, but there are more details...
(没写完还)

### Evaulate A Series Of Video

To evaluate this program on a series of videos, you can simply run `evaluate.py` :

```sh
python evaluate.py --videos test_video.json
```

where `test_video.json` places a series of videos and masks (if provided). It should be formatted like:

```json
{
    "true_positive":[[video1,mask1],[video2,mask2],...,[video_n,mask_n]],
    "true_negative":[[video1,mask1],[video2,mask2],...,[video_n,mask_n]],
    "false_positive":[[video1,mask1],[video2,mask2],...,[video_n,mask_n]]
}
```

If a video has no corresponding mask, simply use `""` .

## Package python codes to executables

In order to successfully freeze MetDetPy programs into stand-alone executables, we suggest using `pyinstaller>=5.0`. You should have `Python>=3.7` installed to avoid compatibility issues. Besides, the package `opencv-python<=4.5.3.56` is required to avoid recursion errors. (not fixed yet, 2022.08.08)

When everything is ready, run `pyinstaller core.spec --clean` to package the code. The target executable will be generated in "./dist/" directory.

## Todo List

 1. 改善对于实际低帧率视频的检测效果 (Almost Done, but some potential bugs left)
 2. 完善日志系统
 3. 改善对蝙蝠等情况的误检
 4. 支持rtmp
 5. 添加GUI
 6. 为不同信噪比/焦距的图像设置合适的超参数组合？(优先？新Detector)
 7. 添加使用分辨率的接口
 8. 支持导出UFO Analizer格式的文本，用于流星组网联测等需求
 9. 自动启停
 10. 时间水印

## Appendix

### Special Thanks

[奔跑的龟斯](https://weibo.com/u/1184392917)
[MG_Raiden扬](https://weibo.com/811151123)
[DustYe夜尘](https://space.bilibili.com/343640654)
星北之羽
纸片儿
LittleQ

### Done

 1. Improving non-ASCII filename support for packaging version /改善对非ASCII文字路径的支持
    By packaging with Python 3.7 and later, the new `pyinstaller` (>=5) can support non-ASCII paths and filenames.

### Performance and Efficiency

 1. With `MergeStacker`, MetDetPy now can detect meteors with a 20% time cost of video length on average (tested with an Intel i5-7500).

 2. Test tool `evaluate.py` is going to be updated soon. For now, MetDetPy performs great for videos from monitoring cameras. For camera-captured videos, the ratio of false positive samples still seems to be a little high.
  