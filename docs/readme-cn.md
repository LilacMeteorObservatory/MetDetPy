# MetDetPy

其他语言版本：[[Eng]](../readme.md)

MetDetPy 是一个基于 python 开发的视频流星检测项目，可用于从直录视频中检测流星。

* MetDetPy受到[uzanka/MeteorDetector](https://github.com/uzanka/MeteorDetector)项目的启发。本项目中也复现了该工作。

* 开发了M3检测器，适用于单帧曝光时间在1/120s-1/4s的流星直录视频。其通过在更宽的时间窗口中有效率的计算差值帧以改善检测性能。

* 设计了一种自适应二值化阈值算法，根据视频的信噪比动态选择二值化阈值（实验性功能）。

* 开发了用于流星检测结果管理器（MeteorLib），用于整合预测，排除假阳性样本。每个预测会被给予一个置信得分，代表其为流星的概率。

* 未来将提供用于测试和评估的脚本，用于选择最佳阈值和检测器。

## 发行版

你可以从[Release](https://github.com/LilacMeteorObservatory/MetDetPy/releases)处获取最新的MetDetPy发行版。发行版将MetDetPy进行了打包，可独立在主流平台运行（Windows，macOS及Linux）。你也可以自行使用 `pyinstaller` 或 `nuitka` 构建独立的可执行文件（见 [打包Python代码为可执行程序](#打包Python代码为可执行程序))。

此外，MetDetPy 从 Meteor Master 的 1.2.0 版本开始作为其后端。Meteor Master是由 [奔跑的龟斯](https://www.photohelper.cn) 开发的视频流星检测软件，在MetDetPy的基础上提供了完善的GUI，多种直播流支持，便捷的导出和自动启停等功能。Meteor Master的最新版本可以从如下源获取:

* [Meteor Master官方网站](https://www.photohelper.cn/MeteorMaster)
* [百度网盘](https://pan.baidu.com/s/1B-O8h4DT89y_u1_YKXKGhA) (Access Code: jz01)


## 运行需求

### 环境

* Python>=3.7

### 模块

* numpy>=1.15.0
* opencv_python>=4.7.0
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

* --cfg: 配置文件。默认情况下使用同目录下的[config.json](../config.json)文件。

* --mask：指定掩模（遮罩）图像。可以使用任何非白色颜色的颜色在空白图像上覆盖不需要检测的区域来创建掩馍图像。不强制要求尺寸与原图相同。支持JPEG和PNG格式。

* --start-time：检测的开始时间。单位为ms。默认从头开始分析。

* --end-time：检测的结束时间。单位为ms。不指定将分析到视频结尾。

* --mode：运行模式。从{backend, frontend}中选择。frontend运行时会显示运行相关信息的进度条，backend则具有随时刷新的输出流，适合作为后端时使用管道进行输出。默认情况下使用frontend。

* --debug: 调试模式。以调试模式启动MetDetPy时，会创建一个额外调试窗口，显示当前的检测情况以及更多调试信息。

#### 覆盖参数

以下参数在不设置时使用[配置文件](../config.json)中的默认值，如果设置则覆盖配置文件中的数值。有关这些参数的详细解释可以参考[配置文件文档](./config-doc.md)。

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

### 评估（即将上线）

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

### 其他工具的使用

#### ClipToolkit

ClipToolkit可用于一次性创建一个视频中的多段视频切片或这些视频段的堆栈图像。其用法如下：

```sh
python ClipToolkit.py [--mode {image,video}] [--suffix SUFFIX] [--save-path SAVE_PATH] [--resize RESIZE] [--jpg-quality JPG_QUALITY] [--png-compressing PNG_COMPRESSING] target json
```

* target: 目标视频文件。
* --mode: convert clip(s) to images or videos. Should be selected from {image, video}.
* --suffix: the suffix of the output. By default, it is "jpg" for image mode and "avi" for video mode.
* --save-path: the path where image(s)/video(s) are placed.

* json: JSON格式的字符串或者JSON文件的路径。该JSON应当包含起始时间，结束时间和文件名（可选）信息。
    具体来说，这个 JSON 应该是一个数组，其中每个元素都应该至少包含一个`"time"`键，其值应是两个`"hh:mm:ss.ms"`格式的字符串组成的数组，表示片段的开始时间和结束时间。 `"filename"` 是一个可选键，您可以在其值中指定文件名和后缀（即视频剪辑应该转换为何种格式并命名。）`"filename"` 优先于 `--mode` 和 ` --suffix` 选项，但如果未指定，此剪辑将根据命令选项自动转换和命名。
    我们提供 [clip_test.json](../test/clip_test.json) 作为用例及测试用 JSON。

* --mode：将剪辑转换为图像或视频。 应从 {image, video} 中选择。 此选项将由 json 中的特定文件名覆盖。

* --suffix：输出的后缀。 默认情况下，图像模式为“jpg”，视频模式为“avi”。 此选项将由 JSON 中的特定文件名覆盖。

* --save-path：放置图像/视频的路径。 当 JSON 中只包含一个片段时，您可以在 --save-path 中包含文件名以简化您的 JSON。

* --resize：将图像/视频调整为给定的分辨率。

* --png-compressing: 生成的png图像压缩程度。 其值应为$Z \in [0,9]$； 默认情况下取值为3。

* --jpg-quality: 生成的jpg图像的质量。 其值应为$Z \in [0,100]$； 默认情况下取值为95。

一个典型的使用例如下:

```sh
python ./ClipToolkit.py ./test/20220413Red.mp4 ./test/clip_test.json --mode image --suffix jpg --jpg-quality 60 --resize 960x540 --save-path ./test
```

注意：如果使用 JSON 格式的字符串而不是 JSON 文件的路径，你应该注意命令行中双引号的转义。

## 打包Python代码为可执行程序

我们提供了 [make_package.py](../make_package.py) 来将MetDetPy打包为独立的可执行程序。该工具支持使用 `pyinstaller` 或 `nuitka` 来打包/编译。

当使用该脚本时，请确保至少安装了`pyinstaller` 或 `nuitka` 中的任意一个工具。此外，在使用 `nuitka` 作为编译工具时，请确保在您的设备上有至少一个C/C++编译器可用。

该工具的用法如下：

```sh
python make_package.py [--tool {nuitka,pyinstaller}] [--mingw64]
     [--apply-upx] [--apply-zip] [--version VERSION]
```

* --tool: your compile/package tool. It should be selected from {nuitka,pyinstaller}. `nuitka` is the default option.

* --mingw64: use the mingw64 compiler. Only worked when using `nuitka` and your OS is windows.

* --apply-upx: apply UPX to squeeze the size of the executable program. Only worked when using `nuitka`.

* --apply-zip: generate zip package when compiling/packaging is finished.

* --version: MetDetPy version tag. Used for naming zip package.


就绪之后，运行 `pyinstaller core.spec --clean` 以打包代码。目标可执行程序会生成在 [dist](../dist/) 目录下。

注意：

1. 建议使用 `Python>=3.9`, 且安装 `pyinstaller>=5.0` 或 `nuitka>=1.3.0` 以避免兼容性问题。

2. 根据在 MetDetPy 上的测试，`pyinstaller` 打包更快，能生成更小的可执行文件（小于Nuitka约30%）。然而，其可执行程序在启动会花更多时间。相对的，`nuitka`花费更多时间在编译期，并生成相对更大的可执行文件（即使使用UPX压缩），但其启动快于Pyinstaller版本约50%。除去启动时间，两种可执行文件运行速度基本相同。因此，可根据实际需求选择合适的打包工具。

3. 由于Python的特性，上述两种工具均无法跨平台打包生成可执行文件。

## Todo List

 1. 改善对于实际低帧率视频的检测效果 (Almost Done, but some potential bugs left)
    1. 找到合适的超参数： max_gap
    2. 设计再校验机制：利用叠图结果做重校准
    3. 优化速度计算逻辑，包括方向，平均速度等
    4. 改善自适应阈值：当误检测点很多时，适当提高分割阈值
 2. 改善对蝙蝠/云等情况的误检(!!)
 3. 支持导出UFO Analizer格式的文本，用于流星组网联测等需求
 4. 快速叠图
 5. 评估系统
 6. 利用cython改善性能
 7. 添加天区解析功能，为支持快速叠图，组网提供基础支持



P.S: 目前结合MeteorMaster已支持/将支持以下功能，它们在MetDetPy的开发中优先级已下调：

 1. 完善的GUI
 2. 支持rtmp/rtsp/http流直播
 3. 时间水印（待开发）
 4. 自动启停

### 性能和效率

 1. 启用 `MergeStacker` 的情况下，MetDetPy大约平均使用20-30% 的视频时长。(使用 Intel i5-7500 测试。根据视频比特率，帧率会有浮动).

 2. 评估工具 `evaluate.py` 将在近期上线。

## 附录

### 特别鸣谢

uzanka [[Github]](https://github.com/uzanka)

奔跑的龟斯 [[Personal Website]](https://photohelper.cn)[[Weibo]](https://weibo.com/u/1184392917)

纸片儿 [[Github]](https://github.com/ArtisticZhao)

DustYe夜尘[[Bilibili]](https://space.bilibili.com/343640654)

RoyalK[[Weibo]](https://weibo.com/u/2244860993)[[Bilibili]](https://space.bilibili.com/259900185)

MG_Raiden扬[[Weibo]](https://weibo.com/811151123)[[Bilibili]](https://space.bilibili.com/11282636)

星北之羽[[Bilibili]](https://space.bilibili.com/366525868/)

LittleQ

韩雅南

来自偶然

ylp

### 更新日志

见 [更新日志](update-log.md)。