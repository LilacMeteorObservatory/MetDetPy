# MetDetPy

MetDetPy is a python script project that can detect meteors from video files.

MetDetPy is enlightened by [MeteorDetector](https://github.com/uzanka/MeteorDetector). Different from their frame-difference approach, we apply a shifted window highlight detection (by misusing max to median) to achieve highly sensitive detection.

## Requirements

Python>=3.6

    numpy>=1.15.0
    opencv_python>=4.0.0
    tqdm>=4.0.0

## Usage

### Run Detection Program Directly

```sh
python core.py  [--cfg CFG] [--mask MASK] 
                [--mode {backend,frontend}] 
                [--debug-mode DEBUG_MODE] 
                target
```

#### Configs
    --cfg: 指定配置文件。默认情况下使用同目录下的config.json文件。
    --mask：指定掩模（遮罩）图像。使用黑色（任何颜色）涂抹不需要检测的区域即可。不强制要求尺寸与原图相同。默认不使用。
    --mode：指定以前台方式运行（即直接启动）或作为后端被调用。两种情况下的输出流会存在一定差异。
    target: 待检测视频。目前主要支持H264编码视频。

#### Example

```sh
```

### Evaulate A Series Of Video




## TODO:

 1. 更快的速度
 2. 完善日志系统
 3. 改善对蝙蝠等情况的误检
 4. 改善对中文路径的支持
 5. 支持rtmp
 6. 添加GUI