# MetDetPy

其他语言版本：[[Eng]](../readme.md)

MetDetPy 是一个基于 python 的视频流星检测项目，可用于从视频中检测流星。

* MetDetPy受到[uzanka/MeteorDetector](https://github.com/uzanka/MeteorDetector)项目的启发，并使用Python3复现了该工作。

* 开发了M3检测器，适用于单帧曝光时间在1/120s-1/4s的流星直录视频。其通过在更宽的时间窗口中有效率的计算差值帧以改善检测性能。

* 设计了一种自适应二值化阈值算法，根据视频的信噪比动态选择二值化阈值（实验性功能）。

* 开发了用于流星检测结果管理器（MeteorLib），用于整合预测，排除假阳性样本。每个预测会被给予一个置信得分，代表其为流星的概率。

* 未来将提供用于测试和评估的脚本，用于选择最佳阈值和检测器。

## 发行版

MetDetPy 从 MeteorMaster 的 1.2.0 版本开始作为其后端。MeteorMaster的Windows发行版可以从如下源获取:

* [Photohelper.cn](https://www.photohelper.cn/MeteorMaster)
* [Baidu NetDisk](https://pan.baidu.com/s/1B-O8h4DT89y_u1_YKXKGhA) (Access Code: jz01)

目前 MetDetPy 没有直接的发行版，但这已在将来的计划中。在那之前，你可以使用`pyinstaller`构建MetDetPy的可执行文件（见 [打包Python代码为可执行程序](#打包Python代码为可执行程序)).

## 运行需求

### 环境

* Python>=3.6

### 模块

* numpy>=1.15.0
* opencv_python>=4.0.0
* tqdm>=4.0.0

可以通过如下指令安装模块：

```sh
pip install -r requirements.txt
```

## 用法

### 直接运行

```sh
python core.py target [--cfg CFG] [--mask MASK] [--start-time START_TIME] [--end-time END_TIME] 
               [--exp-time EXP_TIME] [--mode {backend,frontend}] [--debug]
               [--resize RESIZE] [--adaptive-thre ADAPTIVE_THRE] [--bi-thre BI_THRE | --sensitivity SENSITIVITY]
```

#### 主要参数

* target: 待检测视频文件。支持常见的视频编码。

* --cfg: 配置文件。默认情况下使用同目录下的config.json文件。

* --mask：指定掩模（遮罩）图像。可以使用任何非白色颜色的颜色在空白图像上覆盖不需要检测的区域来创建掩馍图像。不强制要求尺寸与原图相同。支持JPEG和PNG格式。

* --start-time：检测的开始时间。单位为ms。默认从头开始分析。

* --end-time：检测的结束时间。单位为ms。不指定将分析到视频结尾。

* --mode：运行模式。从{backend, frontend}中选择。frontend运行时会显示运行相关信息的进度条，backend则具有随时刷新的输出流，适合作为后端时使用管道进行输出。默认情况下使用frontend。

* --debug: 调试模式。以调试模式启动MetDetPy时，会创建一个额外调试窗口，显示当前的检测情况以及更多调试信息。

#### 覆盖参数

以下参数在不设置时使用配置文件中的默认值，如果设置则覆盖配置文件中的数值。有关这些参数的详细解释可以参考[配置文件文档](./config-doc.md)。

* --resize: 检测时采用的帧图像尺寸。

* --exp-time: 单帧曝光时间。可用一个浮点数或从 {auto, real-time, slow} 中选择一项以指定。大多数情况下可以使用 "auto"。

* --adaptive-thre: 检测器中是否启用自适应二值化阈值。从{on, off}中选择。

* --bi-thre: 指定检测器中使用的二值化阈值。当启用自适应二值化阈值时，该选项无效化。不能与--sensitivity同时设置。

* --sensitivity: 检测器的灵敏度。从{low, normal, high}中选择。当启用自适应二值化阈值时，灵敏度选项仍起效，且更高的灵敏度将会估计更高的阈值。不能与--bi-thre同时设置。

#### 示例

```sh
python core.py ./test/20220413Red.mp4 --mask ./test/mask-east.jpg
```

#### 自定义配置文件

大多数与检测相关的重要参数都预先定义并储存在配置文件中。大多数情况下，这些预设值都能够较好的工作，但有时对参数微调可以取得更好的结果。如果想要获取配置文件的说明，可以参考[配置文件文档](./config-doc.md)获取更多信息。

### 评估

若需要评估MetDetPy在某个视频上的检测性能，可以运行 `evaluate.py` :

```sh
python evaluate.py --videos test_video.json
```

其中， `test_video.json` 应包含有关测试视频中所有流星的位置和时间标注，并按照以下形式构建为json：

```json
{
    "video": "path/to/the/video.mp4",
    "mask": "path/to/the/mask.jpg",
    "gt": [{
        "start_time": "00:00:03.750000",
        "end_time": "00:00:04.333000",
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

若没有对应的mask，可以用 `""` 表示。

## 打包Python代码为可执行程序

为了能够成功冻结MetDetPy为独立的可执行程序，我们推荐使用`pyinstaller>=5.0`。为避免兼容性问题，最好使用 `Python>=3.7`。此外，为了避免递归错误，最好使用 `opencv-python<=4.5.3.56`。

就绪之后，运行 `pyinstaller core.spec --clean` 以打包代码。目标可执行程序会生成在 [dist](./dist/) 目录下。

## Todo List

 1. 改善对于实际低帧率视频的检测效果 (Almost Done, but some potential bugs left)
    1. 找到合适的超参数： max_gap
    2. 再校验机制
    3. 优化速度计算逻辑，包括方向，平均速度等
    4. 改善自适应阈值：当误检测点很多时，适当提高分割阈值
 2. 改善对蝙蝠/云等情况的误检(!!)
 3. 完善日志系统
 4. 支持rtmp
 5. 添加GUI
 6. 支持导出UFO Analizer格式的文本，用于流星组网联测等需求
 7. 自动启停
 8. 时间水印

### 性能和效率

 1. 启用 `MergeStacker` 的情况下，MetDetPy大约平均使用20-30% 的视频时长。(使用 Intel i5-7500 测试。根据视频比特率，帧率会有浮动).

 2. 评估工具 `evaluate.py` 将在近期上线。

## 附录

### 特别鸣谢

[uzanka](https://github.com/uzanka)

[奔跑的龟斯](https://weibo.com/u/1184392917)

[纸片儿](https://github.com/ArtisticZhao)

[DustYe夜尘](https://space.bilibili.com/343640654)

[RoyalK](https://weibo.com/u/2244860993)

[MG_Raiden扬](https://weibo.com/811151123)

[星北之羽](https://space.bilibili.com/366525868/)

LittleQ

### 更新日志

#### Version 1.2

✅ 优化了分辨率相关接口以支持不同长宽比的视频

✅ 实现根据方差计算的自适应二值化阈值

✅ 增加灵敏度设置

✅ 输出调整为概率形式

#### Version 1.1.1

✅ 改善对非ASCII文字路径的支持

#### Version 1.1

✅ 增加了后端模式
