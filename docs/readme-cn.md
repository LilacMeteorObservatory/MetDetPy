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

## 快速开始

MetDetPy 设计有合理的默认配置，在多数情况下无需额外参数即可直接运行。

### 视频检测

```sh
# 检测视频（结果仅在命令行输出）
python MetDetPy.py video.mp4

# 检测并保存结果
python MetDetPy.py video.mp4 --save-path results.json

# 检测后，过滤掉所有的负样本和小尺寸结果，仅导出所有流星片段为视频
python MetDetPy.py video.mp4 --save-path results.json
python ClipToolkit.py results.json --mode video --enable-filter-rules --save-path ./output

# 或导出带标注框的图像
python ClipToolkit.py results.json --mode video --enable-filter-rules --with-bbox --save-path ./output
```

### 图像检测

```sh
# 检测图像文件夹
python MetDetPhoto.py ./images

# 检测并保存结果
python MetDetPhoto.py ./images --save-path results.json
```

### 完整流程示例

```sh
# 1. 检测视频，保存结果
python MetDetPy.py video.mp4 --save-path results.json

# 2. 导出所有检测到的流星片段为视频，并在目标上附加标注框
python ClipToolkit.py results.json --mode video --with-bbox  --save-path ./output

# 3. 或导出堆栈图像
python ClipToolkit.py results.json --mode image --save-path ./output
```

## 详细文档

### 工具使用指南

* [检测工具使用指南](./tool-usage/Detector-usage-cn.md) - MetDetPy（视频检测）和 MetDetPhoto（图像检测）
* [ClipToolkit 使用指南](./tool-usage/ClipToolkit-usage-cn.md) - 视频切片和图像堆叠工具
* [Evaluate 工具说明](./tool-usage-cn.md#evaluate) - 性能评估和回归测试工具
* [make_package 工具说明](./tool-usage-cn.md#make-package) - 打包脚本

### 配置和数据格式

* [配置文件说明](./config-doc-cn.md) - 了解各个配置选项的含义
* [数据格式说明](./data-format-cn.md) - 了解输入配置和输出文件格式

## 性能和效率

1. 在 3840x2160 10fps 视频上应用默认配置进行检测时，MetDetPy 检测流星的平均时间开销为视频长度的 20-30%（使用 Intel i5-7500 测试）。 FPS 较高的视频可能会花费更多时间。

2. 我们使用从各种设备（从改装监控摄像头到数码相机）拍摄的样本视频测试 MetDetPy，MetDetPy 平均能够达到 80% 以上的准确率和 80% 以上的召回率。

3. MetDetPy 现在可以快速高效地检测大多数流星视频。 但当面对复杂的天气或其他影响因素时，其准确率和召回率还有待提高。 如果您发现 MetDetPy 在您的视频上表现不够好，欢迎联系我们或提交问题（如果可以的话，一并提供完整或剪辑的视频）。

## 许可

该项目根据 Mozilla 公共许可证 2.0 (MPL-2.0) 获得许可。这意味着您可以自由使用、修改和分发该软件，但须满足以下条件：

* 源代码可用性：您对源代码所做的任何修改也必须在 MPL-2.0 许可证下可用。
* 文件级 Copyleft：您可以将此软件与不同许可证下的其他代码结合使用，但对 MPL-2.0 许可文件的任何修改都必须保留在同一许可证下。
* 无保证：软件按"原样"提供，不提供任何形式的明示或暗示的保证。使用风险自负。

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
