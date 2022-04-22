# MetDetPy

MetDetPy is a python script project that can detect meteors from video files.

MetDetPy is enlightened by [MeteorDetector](https://github.com/uzanka/MeteorDetector). Different from their frame-difference approach, we apply a shifted window highlight detection (by misusing max to median) to achieve highly sensitive detection.

## Requirements

    numpy>=1.15.0
    opencv_python>=4.0.0
    tqdm>=4.0.0

## Usage

```sh
python core.py  [--cfg CFG] [--mask MASK] [--mode {backend,frontend}]
[--debug-mode DEBUG_MODE]
               target
```




## TODO:

 1. 更快的速度
 2. 完善日志系统
 3. 改善对蝙蝠等情况的误检
 4. 改进对中文路径的支持
 5. 添加GUI