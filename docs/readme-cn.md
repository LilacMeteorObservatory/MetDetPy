  <div align="center">
  <img src="../imgs/banner.png"/>

[![GitHub release](https://img.shields.io/github/release/LilacMeteorObservatory/MetDetPy.svg)](https://github.com/LilacMeteorObservatory/MetDetPy/releases/latest) [![GitHub Release Date](https://img.shields.io/github/release-date/LilacMeteorObservatory/MetDetPy.svg)](https://github.com/LilacMeteorObservatory/MetDetPy/releases/latest) [![license](https://img.shields.io/github/license/LilacMeteorObservatory/MetDetPy)](./LICENSE) [![Github All Releases](https://img.shields.io/github/downloads/LilacMeteorObservatory/MetDetPy/total.svg)](https://github.com/LilacMeteorObservatory/MetDetPy/releases)

<center>语言: <a href="../readme.md">English</a> | 简体中文 </center>

</div>

## 简介

MetDetPy 可从直录视频或图像中检测流星。其视频检测受到[uzanka/MeteorDetector](https://github.com/uzanka/MeteorDetector)项目的启发。MetDetPy 具有以下优点：

* **易于使用且可配置：** MetDetPy 设计有易用的默认配置，多数情况下无需详细配置参数即可以进行高效的流星检测，并且也支持修改设置以获得更好的检测结果。

* **适用于各种设备和曝光时间：** MetDetPy 可以从各种设备拍摄的视频及图像中检测流星。借助一系列自适应算法与深度学习模型，MetDetPy对从流星监控到数码相机拍摄的数据都能较好工作。

* **深度学习模型接入：** MetDetPy 可以在主检测或重校验阶段选择性使用深度学习模型，在不显著增加性能开销的情况下提升检测效果。模型也可用于图像中的流星检测。

* **有效的过滤器：** 流星结果将根据其视觉特性与运动属性进行重校验以排除误报样本。每个预测都将被给出一个取值范围为 `[0,1]` 的置信度分数，表示其被认为是流星的可能性。

* **丰富的支持工具：** MetDetPy 还提供了数个工具以支持评估和导出功能，包括一个评估工具和剪辑工具，以支持进行高效的视频切片、图像堆叠或结果评估。

## 发行版

你可以从 [Release](https://github.com/LilacMeteorObservatory/MetDetPy/releases) 处获取最新的MetDetPy发行版。发行版将 MetDetPy 进行了打包，可独立在主流平台运行（Windows，macOS）。你也可以自行使用 `nuitka` 构建独立的可执行文件（见 [打包Python代码为可执行程序](./tool-usage-cn.md#打包Python代码为可执行程序))。

此外，MetDetPy 从 Meteor Master 的 1.2.0 版本开始作为其后端。Meteor Master (AI)是由 [奔跑的龟斯](https://www.photohelper.cn) 开发的流星检测软件，在 MetDetPy 的基础上提供了完善的GUI，多种直播流支持，便捷的导出和自动启停等功能。可以从 [Meteor Master官方网站](https://www.photohelper.cn/MeteorMaster) 获取更多信息，或从微软商店/App Store获取其最新版。其早期版本可从 [百度网盘](https://pan.baidu.com/s/1B-O8h4DT89y_u1_YKXKGhA) (Access Code: jz01) 获取。

## 运行需求

### 环境

* 64bit OS
* Python>=3.7 (推荐 3.9+)

### Python 依赖

* numpy>=1.15.0
* opencv_python>=4.9.0
* tqdm>=4.0.0
* multiprocess>=0.70.0
* onnxruntime>=1.16.0
* av>=15.0.0
* dacite>=1.9.0
* pyexiv2>=2.12.0

可以通过如下指令安装依赖：

```sh
pip install -r requirements.txt
```

### GPU 支持

上述软件包能够使 MetDetPy 正常运行，但深度学习模型仅支持在 CPU 设备上运行。如果希望利用 GPU，可以按照以下方式额外安装或替换 onnxruntime 相关库：

* **Windows/Linux 用户（推荐）：** 如果您使用的是 Windows 或 Linux，建议额外安装 `onnxruntime_directml`。该库利用 DirectX 进行模型推理加速，适用于大多数 GPU（包括 Nvidia、AMD、Intel 等厂商的显卡）。

* **Nvidia GPU 用户（高级）：** 如果您使用的是 Nvidia GPU 并且已安装 CUDA，可以安装与 CUDA 版本匹配的 `onnxruntime-gpu` 代替 `onnxruntime`。这可以启用 CUDA 加速，从而带来更高的性能。

#### ⚠️ 注意

* 对于 macOs 用户，CoreML模型推理加速已经被集成到 `onnxruntime` 中，无需额外配置即可启用GPU支持。

* 在当前发布版本中，Windows 软件包使用 `onnxruntime_directml`。默认的 CUDA 支持将在准备好后添加。

## 用法

### 视频流星检测

`MetDetPy` 是视频流星检测的启动器，其用法如下：

```sh
python MetDetPy.py target [--cfg CFG] [--mask MASK] [--start-time START_TIME] [--end-time END_TIME] 
               [--exp-time EXP_TIME] [--mode {backend,frontend}] [--debug]
               [--resize RESIZE] [--adaptive-thre ADAPTIVE_THRE] [--bi-thre BI_THRE | --sensitivity SENSITIVITY]
               [--recheck RECHECK] [--save-rechecked-img SAVE_RECHECKED_IMG]
               [--provider {cpu,default,coreml,dml,cuda}][--live-mode {on,off}][--save-path SAVE-PATH]
```

#### 主要参数

* `target`: 待检测视频文件。支持常见的视频编码。

* `--cfg`: 配置文件。默认情况下使用config目录下的[m3det_normal.json](../config/m3det_normal.json)文件。

* `--mask`：指定掩模（遮罩）图像。可以使用任何非白色颜色的颜色在空白图像上覆盖不需要检测的区域来创建掩馍图像。不强制要求尺寸与原图相同。支持JPEG和PNG格式。

* `--start-time`：检测的开始时间。可以输入单位为ms的整数或是形如`"HH:MM:SS"`的字符串。默认从头开始分析。

* `--end-time`：检测的结束时间。可以输入单位为ms的整数或是形如`"HH:MM:SS"`的字符串。不指定将分析到视频结尾。

* `--mode`：运行模式。从 `{backend, frontend}` 中选择。`frontend` 运行时会显示运行相关信息的进度条，backend 则具有随时刷新的输出流，适合作为后端时使用管道进行输出。默认情况下使用 `frontend`。

* `--debug`: 调试模式。调试模式启动时会打印更详细的日志信息。

* `--visu`: 可视模式。以可视模式启动时，会创建一个额外视频窗口，显示当前的检测情况。

* `--live-mode`: 实时模式。当在实时模式下运行时，检测的运行时间将会被控制在尽可能接近实际视频时长的程度，这可以帮助均衡CPU的开销。 从 `{on, off}` 中选择。

* `--provider`: 指定优先使用的模型后端，可选值会根据平台有所差异。如果指定的模型后端不可用，则会使用默认选项。

* `--save-path`: 保存[MDRF](./tool-usage.md#meteor-detection-recording-format-mdrf)格式的检测结果到给定的路径或JSON文件下。

#### 额外参数

以下参数在不设置时使用配置文件中的默认值，如果设置则覆盖配置文件中的数值。有关这些参数的详细解释可以参考[配置文件文档](./config-doc-cn.md)。

* `--resize`: 检测时采用的帧图像尺寸。可以指定单个整数（如`960`，代表长边长度），列表（如`[960,540]`）或者字符串（如`960x540`）。

* `--exp-time`: 单帧曝光时间。可用一个浮点数或从 `{auto, real-time, slow}` 中选择一项以指定。大多数情况下可以使用 `"auto"`。

* `--adaptive-thre`: 检测器中是否启用自适应二值化阈值。从{on, off}中选择。

* `--bi-thre`: 指定检测器中使用的二值化阈值。当启用自适应二值化阈值时，该选项无效化。不能与`--sensitivity`同时设置。

* `--sensitivity`: 检测器的灵敏度。从 `{low, normal, high}` 中选择。当启用自适应二值化阈值时，灵敏度选项仍起效，且更高的灵敏度将会估计更高的阈值。不能与 `--bi-thre` 同时设置。

* `--recheck`: 启用重校验机制减少误报。从`{on, off}`中选择。

#### 示例

```sh
python MetDetPy.py "./test/20220413Red.mp4" --mask "./test/mask-east.jpg" --visu --save-path .
```

#### 输出

`MetDetPy` 将检测结果输出于命令行，以便在运行中实时确认检测情况。如果希望使用检测结果生成其他内容，例如生成流星片段，流星截图，标注文件等，最佳实践是在运行时指定 `--save-path`参数，将检测结果保存到 `MDRF` 格式的文件中。保存的结果可以被`MetDetPy`项目中的[其他工具](#其他工具的使用)处理。输出的结果格式见[数据格式](./data-format-cn.md)页。


#### 自定义配置文件

大多数与检测相关的重要参数都预先定义并储存在配置文件中。大多数情况下，这些预设值都能够较好的工作，但有时对参数微调可以取得更好的结果。如果想要获取配置文件的说明，可以参考[配置文件文档](./config-doc-cn.md)获取更多信息。

### 图像流星检测

`MetDetPhoto` 是图像流星检测的启动器，其用法如下：

```sh
python MetDetPhoto.py target [--mask MASK]
                             [--model-path MODEL_PATH] [--model-type MODEL_TYPE] 
                             [--exclude-noise] [--debayer] [--debayer-pattern DEBAYER_PATTERN]
                             [--visu] [--visu-resolution VISU_RESOLUTION]
                             [--save-path SAVE_PATH]
```

#### 参数

* `target`: 目标图像文件/文件夹。支持单张图像，图像文件夹以及常规视频编码格式的延时视频文件。

* `--mask`: 掩模图像。可以使用任何非白色颜色的颜色在空白图像上覆盖不需要检测的区域来创建掩馍图像。不强制要求尺寸与原图相同。支持JPEG和PNG格式。

* `--model-path`: 模型权重文件路径。默认使用[./weights/yolov5s_v2.onnx](./weights/yolov5s_v2.onnx)权重。

* `--model-type`: 模型格式，决定如何处理模型的输出。目前仅支持 `YOLO` 格式。默认值为 `YOLO`。

* `--exclude-noise`: 输出时从预测排除常见噪声类型（如卫星和飞虫），仅保存正样本到文件。

* `--debayer`: 是否在处理延时视频前对视频帧进行Debayer变换。

* `--debayer-pattern`: Debayer使用的矩阵，如 RGGB 或 BGGR。仅在 `--debayer` 选项启用时生效。

* `--visu`: 可视模式。以可视模式启动时，会创建一个额外视频窗口，显示当前的检测情况。

* `--visu-resolution`: 可视化窗口的分辨率设置。

* `--save-path`: 保存检测结果到 [MDRF](./tool-usage.md#meteor-detection-recording-format-mdrf) 格式文件中。

#### 示例

```sh
python MetDetPhoto.py "/path/to/your/folder" --mask "/path/to/your/mask.jpg" --exclude-noise --save-path .
```

### 其他工具的使用

MetDetPy还提供了数个工具以支持评估和导出功能，包括Evaluate（效果评估和回归测试工具）、ClipToolkit （批图像堆栈和视频剪辑工具）和make_package（打包脚本）。访问[工具文档](./tool-usage-cn.md)了解这些工具的使用方法。

可以组合使用工具，实现从检测到结果导出。例如，如果希望检测一个视频中所有的流星，并得到所有的视频片段对应的最大值叠加图像，保存于当前目录下，可按顺序执行以下命令：

```sh
# 检测视频，输出 20220413Red.json 于当前目录下
python MetDetPy.py "./test/20220413Red.mp4" --mask "./test/mask-east.jpg" --visu --save-path .
# 导出结果到当前目录
python ClipToolkit.py 20220413Red.json --mode image --save-path .
```

### 数据规范

要了解各个工具的输入配置以及输出文件格式，请参阅[数据规范](./data-format-cn.md)文档。

## 性能和效率

1. 在 3840x2160 10fps 视频上应用默认配置进行检测时，MetDetPy 检测流星的平均时间开销为视频长度的 20-30%（使用 Intel i5-7500 测试）。 FPS 较高的视频可能会花费更多时间。

2. 我们使用从各种设备（从改装监控摄像头到数码相机）拍摄的样本视频测试 MetDetPy，MetDetPy 平均能够达到 80% 以上的准确率和 80% 以上的召回率。

3. MetDetPy 现在可以快速高效地检测大多数流星视频。 但当面对复杂的天气或其他影响因素时，其准确率和召回率还有待提高。 如果您发现 MetDetPy 在您的视频上表现不够好，欢迎联系我们或提交问题（如果可以的话，一并提供完整或剪辑的视频）。

## 许可

该项目根据 Mozilla 公共许可证 2.0 (MPL-2.0) 获得许可。这意味着您可以自由使用、修改和分发该软件，但须满足以下条件：

* 源代码可用性：您对源代码所做的任何修改也必须在 MPL-2.0 许可证下可用。
* 文件级 Copyleft：您可以将此软件与不同许可证下的其他代码结合使用，但对 MPL-2.0 许可文件的任何修改都必须保留在同一许可证下。
* 无保证：软件按“原样”提供，不提供任何形式的明示或暗示的保证。使用风险自负。

欲了解更多详细信息，请参阅[MPL-2.0许可证](https://www.mozilla.org/en-US/MPL/2.0/)。

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

### 更新日志 / TODO列表

见 [更新日志](./update-log.md)。

### 统计数据

[![Star History Chart](https://api.star-history.com/svg?repos=LilacMeteorObservatory/MetDetPy&type=Timeline)](https://star-history.com/#LilacMeteorObservatory/MetDetPy&Timeline)