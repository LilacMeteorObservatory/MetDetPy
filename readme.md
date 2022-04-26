# MetDetPy

MetDetPy is a python script project that can detect meteors from video files.

MetDetPy is enlightened by [MeteorDetector](https://github.com/uzanka/MeteorDetector). Different from their frame-difference approach, we apply a shifted window highlight detection (by misusing max to median) to achieve highly sensitive detection.

## Release Version

MetDetPy (is going to) work as the backend of the Meteor Master since version 1.2. You can get its Windows release version from:

* [Photohelper.cn](https://www.photohelper.cn/MeteorMaster)
* [Baidu NetDisk](https://pan.baidu.com/s/1B-O8h4DT89y_u1_YKXKGhA) (Access Code: jz01)

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

### Run Detection Program Directly

```sh
python core.py  [--cfg CFG] [--mask MASK] 
                [--mode {backend,frontend}] 
                [--start-time START_TIME] [--end-time END_TIME]
                [--debug-mode DEBUG_MODE] 
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

## TODO

 1. 更快的速度 Faster Speed (now working on i5-7500 with an average of 50 fps)
 2. 改善对于实际低帧率视频的检测效果
 3. 完善日志系统
 4. 改善对蝙蝠等情况的误检
 5. 改善对中文路径的支持
 6. 支持rtmp
 7. 添加GUI
