<div align="center">
  <img src="../imgs/banner.png"/>

[![license](https://img.shields.io/badge/license-LGPL2.1-success)](./LICENSE)

<center>语言: <a href="../readme.md">English</a> | 简体中文 </center>

</div>

## 简介

MetDetPy 是一个基于 python 开发的，可用于从直录视频中检测流星的检测器。其受到[uzanka/MeteorDetector](https://github.com/uzanka/MeteorDetector)项目的启发。MetDetPy更加强大可靠，具有以下优点：

* **自适应灵敏度：** 对于大多数流星视频，MetDetPy可以借助一系列自适应算法自动估算信噪比并调整灵敏度，无需详细配置参数即可以进行高效的流星检测。

* **适用于各种设备和曝光时间：** MetDetPy可以从各种设备（流星监控，数码相机等）拍摄的视频文件中检测流星。 我们实现了 M3 检测器，它适用于曝光时间从 1/120 秒到 1/4 秒的流星直录视频。 它在更宽的滑动时间窗口中有效地计算差异帧（由最大减去均值计算）以提高准确性。

* **低 CPU 和内存使用率：** MetDetPy 基于 OpenCV 开发的，因此工作时对 CPU 和内存占用率较低，不依赖 GPU。 可支持在主流电脑或准系统上进行多摄像头输入的实时检测。

* **有效的过滤器：** MetDetPy引入了流星检测结果管理器（称为 MeteorLib）以整合预测，排除误报样本。 每个预测都有一个取值范围为 [0,1] 的置信度分数，表示其被认为是流星的可能性。

* **丰富的支持工具：** MetDetPy还提供了评估工具和剪辑工具，以支持进行高效的视频切片、图像堆叠或结果评估。

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
* easydict>=1.0

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

* --start-time：检测的开始时间。可以输入单位为ms的整数或是形如`"HH:MM:SS"`的字符串。默认从头开始分析。

* --end-time：检测的结束时间。可以输入单位为ms的整数或是形如`"HH:MM:SS"`的字符串。不指定将分析到视频结尾。

* --mode：运行模式。从{backend, frontend}中选择。frontend运行时会显示运行相关信息的进度条，backend则具有随时刷新的输出流，适合作为后端时使用管道进行输出。默认情况下使用frontend。

* --debug: 调试模式。以调试模式启动MetDetPy时，会创建一个额外调试窗口，显示当前的检测情况以及更多调试信息。

#### 覆盖参数

以下参数在不设置时使用[配置文件](../config.json)中的默认值，如果设置则覆盖配置文件中的数值。有关这些参数的详细解释可以参考[配置文件文档](./config-doc.md)。

* --resize: 检测时采用的帧图像尺寸。可以指定单个整数（如`960`，代表长边长度），列表（如`[960,540]`）或者字符串（如`960x540`）。

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

### 其他工具的使用

#### ClipToolkit

ClipToolkit可用于一次性创建一个视频中的多段视频切片或这些视频段的堆栈图像。其用法如下：

```sh
python ClipToolkit.py [--mode {image,video}] [--suffix SUFFIX] [--save-path SAVE_PATH] [--resize RESIZE] [--jpg-quality JPG_QUALITY] [--png-compressing PNG_COMPRESSING] target json
```
##### Arguments:

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

#### 评估

若需要评估MetDetPy在某个视频上的检测性能，可以运行 `evaluate.py` :

```sh
python evaluate.py target [--cfg CFG] [--load LOAD] [--save SAVE] [--metrics] [--debug] video_json
```

##### 参数

* video_json：一个JSON文件，里面放置了视频、遮罩的名称和流星标注。 它的格式应该是这样的：

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

如果没有相应的掩码，只需使用`""`。 如果没有流星标注，`"meteors"`也可以忽略。

* --cfg：配置文件。 使用默认相同路径下的[config.json](./config.json)。

* --load: 加载`evaluate.py` 保存的检测结果的文件名。 如果启用该项，`evaluate.py` 将直接加载结果文件，而不运行检测。

* --save：要保存的检测结果的文件名。

* --metrics：计算检测的精度和召回率。 要应用它，必须在 `"video_json"` 的JSON中提供 `"meteors"` 。

* --debug：用这个启动`evaluate.py`时，会有一个调试窗口。

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

1. 建议使用 `Python>=3.9`, 且安装 `pyinstaller>=5.0` 或 `nuitka>=1.3.0` 以避免兼容性问题。此外，避免使用 `nuitka>=1.5.0` (2023.03)，这可能会导致某些设备运行时出现 SystemError。

2. 根据在 MetDetPy 上的测试，`pyinstaller` 打包更快，能生成更小的可执行文件（小于Nuitka约30%）。然而，其可执行程序在启动会花更多时间。相对的，`nuitka`花费更多时间在编译期，并生成相对更大的可执行文件（即使使用UPX压缩），但其启动快于Pyinstaller版本约50%。除去启动时间，两种可执行文件运行速度基本相同。因此，可根据实际需求选择合适的打包工具。

3. 由于Python的特性，上述两种工具均无法跨平台打包生成可执行文件。

4. （原因不明）如果环境中有`matplotlib`或`scipy`，它们很可能会一起打包到最终目录中。 为避免这种情况，建议使用干净的环境进行打包。
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

### 性能和效率

1. 在 3840x2160 10fps 视频上应用默认配置进行检测时，MetDetPy 检测流星的平均时间开销为视频长度的 20-30%（使用 Intel i5-7500 测试）。 FPS 较高的视频可能会花费更多时间。

2、MetDetPy目前没有引入深度学习模型，因此不需要GPU，可以支持主流电脑或准系统上的多摄像头实时检测（可以利用 [MeteorMaster](https://www.photohelper.cn/ MeteorMaster) 实现）。（PS：我们确实计划在未来添加一个简单轻量级的 CNN 分类器----别担心，它不会显着增加 CPU 负载，同时它可以在可能的情况下使用 Nvidia GPU。）

3. 我们使用从各种设备（从改装监控摄像头到数码相机）拍摄的样本视频测试 MetDetPy，MetDetPy 平均能够达到 80% 以上的准确率和 80% 以上的召回率。

4. MetDetPy 现在可以快速高效地检测大多数流星视频。 但当面对复杂的天气或其他影响因素时，其准确率和召回率还有待提高。 如果您发现 MetDetPy 在您的视频上表现不够好，欢迎联系我们或提交问题（如果可以的话，一并提供完整或剪辑的视频）。

## 附录

### 特别鸣谢

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

### 更新日志

见 [更新日志](update-log.md)。