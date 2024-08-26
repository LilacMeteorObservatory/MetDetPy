  <div align="center">
  <img src="../imgs/banner.png"/>

[![GitHub release](https://img.shields.io/github/release/LilacMeteorObservatory/MetDetPy.svg)](https://github.com/LilacMeteorObservatory/MetDetPy/releases/latest) [![GitHub Release Date](https://img.shields.io/github/release-date/LilacMeteorObservatory/MetDetPy.svg)](https://github.com/LilacMeteorObservatory/MetDetPy/releases/latest) [![license](https://img.shields.io/github/license/LilacMeteorObservatory/MetDetPy)](./LICENSE) [![Github All Releases](https://img.shields.io/github/downloads/LilacMeteorObservatory/MetDetPy/total.svg)](https://github.com/LilacMeteorObservatory/MetDetPy/releases)

<center>语言: <a href="../readme.md">English</a> | 简体中文 </center>

</div>

## 简介

MetDetPy 是一个基于 python 开发的，可从直录视频或图像中检测流星的检测器。其视频检测受到[uzanka/MeteorDetector](https://github.com/uzanka/MeteorDetector)项目的启发。MetDetPy具有以下优点：

* **易于使用且可配置：** MetDetPy设计有易用的默认配置，多数情况下无需详细配置参数即可以进行高效的流星检测，并且也支持修改设置以获得更好的检测结果。

* **适用于各种设备和曝光时间：** MetDetPy可以从各种设备拍摄的视频及图像中检测流星。借助一系列自适应算法与深度学习模型，MetDetPy对从流星监控到数码相机拍摄的数据都能较好工作。

* **低 CPU 使用率：** MetDetPy 工作时对 CPU 和内存占用率较低，可支持在主流电脑或准系统上进行多摄像头输入的实时检测。

* **可选的深度学习模型接入：** MetDetPy 已接入深度学习支持，可以在主检测或重校验阶段选择性使用深度学习模型，在不显著增加性能开销的情况下提升检测效果。模型也可用于图像中的流星检测。

* **有效的过滤器：** 流星结果将根据其视觉特性与运动属性进行重校验以排除误报样本。每个预测都将被给出一个取值范围为 [0,1] 的置信度分数，表示其被认为是流星的可能性。

* **丰富的支持工具：** MetDetPy还提供了评估工具和剪辑工具，以支持进行高效的视频切片、图像堆叠或结果评估。

## 发行版

你可以从 [Release](https://github.com/LilacMeteorObservatory/MetDetPy/releases) 处获取最新的MetDetPy发行版。发行版将 MetDetPy 进行了打包，可独立在主流平台运行（Windows，macOS及Linux）。你也可以自行使用 `nuitka` 或 `pyinstaller` 构建独立的可执行文件（见 [打包Python代码为可执行程序](#打包Python代码为可执行程序))。

此外，MetDetPy 从 Meteor Master 的 1.2.0 版本开始作为其后端。Meteor Master (AI)是由 [奔跑的龟斯](https://www.photohelper.cn) 开发的流星检测软件，在MetDetPy的基础上提供了完善的GUI，多种直播流支持，便捷的导出和自动启停等功能。可以从 [Meteor Master官方网站](https://www.photohelper.cn/MeteorMaster) 获取更多信息，或从微软商店/App Store获取其最新版。其早期版本可从 [百度网盘](https://pan.baidu.com/s/1B-O8h4DT89y_u1_YKXKGhA) (Access Code: jz01) 获取。

## 运行需求

### 环境

* 64bit OS
* Python>=3.7 (推荐 3.9+)

### 依赖

* numpy>=1.15.0
* opencv_python>=4.9.0
* tqdm>=4.0.0
* easydict>=1.0
* multiprocess>=0.70.0
* onnxruntime>=1.16.0

可以通过如下指令安装依赖：

```sh
pip install -r requirements.txt
```

### GPU 支持

上述软件包能够使 MetDetPy 正常运行，但深度学习模型仅支持在 CPU 设备上运行。如果希望利用 GPU，可以按照以下方式额外安装或替换 onnxruntime 相关库：

* **Windows/Linux 用户（推荐）：** 如果您使用的是 Windows 或 Linux，建议额外安装 `onnxruntime_directml`。该库利用 DirectX 进行模型推理加速，适用于大多数 GPU（包括 Nvidia、AMD、Intel 等厂商的显卡）。

* **macOS 用户（推荐）：** 如果您使用的是 macOS，建议安装 `onnxruntime-silicon` 代替 `onnxruntime`。该库利用 CoreML 进行模型推理加速。

* **Nvidia GPU 用户（高级）：** 如果您使用的是 Nvidia GPU 并且已安装 CUDA，可以安装与 CUDA 版本匹配的 `onnxruntime-gpu` 代替 `onnxruntime`。这可以启用 CUDA 加速，从而带来更高的性能。

#### ⚠️ 注意
* 安装 `onnxruntime-silicon` 和 `onnxruntime-gpu` 时，需要先卸载 `onnxruntime`。

* 在当前发布版本中，Windows 软件包使用 `onnxruntime_directml`，macOS 软件包使用 `onnxruntime-silicon`。默认的 CUDA 支持将在准备好后添加。

## 用法

### 运行视频流星检测

```sh
python MetDetPy.py target [--cfg CFG] [--mask MASK] [--start-time START_TIME] [--end-time END_TIME] 
               [--exp-time EXP_TIME] [--mode {backend,frontend}] [--debug]
               [--resize RESIZE] [--adaptive-thre ADAPTIVE_THRE] [--bi-thre BI_THRE | --sensitivity SENSITIVITY]
               [--recheck RECHECK] [--save-rechecked-img SAVE_RECHECKED_IMG]
```

#### 主要参数

* target: 待检测视频文件。支持常见的视频编码。

* --cfg: 配置文件。默认情况下使用config目录下的[m3det_normal.json](../config/m3det_normal.json)文件。

* --mask：指定掩模（遮罩）图像。可以使用任何非白色颜色的颜色在空白图像上覆盖不需要检测的区域来创建掩馍图像。不强制要求尺寸与原图相同。支持JPEG和PNG格式。

* --start-time：检测的开始时间。可以输入单位为ms的整数或是形如`"HH:MM:SS"`的字符串。默认从头开始分析。

* --end-time：检测的结束时间。可以输入单位为ms的整数或是形如`"HH:MM:SS"`的字符串。不指定将分析到视频结尾。

* --mode：运行模式。从 {backend, frontend} 中选择。frontend 运行时会显示运行相关信息的进度条，backend 则具有随时刷新的输出流，适合作为后端时使用管道进行输出。默认情况下使用frontend。

* --debug: 调试模式。以调试模式启动 MetDetPy 时，会打印更详细的日志信息用于调试。

* --visu: 可视模式。以可视模式启动时，会创建一个额外视频窗口，显示当前的检测情况。

#### 额外参数

以下参数在不设置时使用配置文件中的默认值，如果设置则覆盖配置文件中的数值。有关这些参数的详细解释可以参考[配置文件文档](./config-doc-cn.md)。

* --resize: 检测时采用的帧图像尺寸。可以指定单个整数（如`960`，代表长边长度），列表（如`[960,540]`）或者字符串（如`960x540`）。

* --exp-time: 单帧曝光时间。可用一个浮点数或从 {auto, real-time, slow} 中选择一项以指定。大多数情况下可以使用 "auto"。

* --adaptive-thre: 检测器中是否启用自适应二值化阈值。从{on, off}中选择。

* --bi-thre: 指定检测器中使用的二值化阈值。当启用自适应二值化阈值时，该选项无效化。不能与--sensitivity同时设置。

* --sensitivity: 检测器的灵敏度。从{low, normal, high}中选择。当启用自适应二值化阈值时，灵敏度选项仍起效，且更高的灵敏度将会估计更高的阈值。不能与--bi-thre同时设置。

* --recheck: 启用重校验机制减少误报。从{on, off}中选择。

* --save-rechecked-img: 重校验图像的保存路径。


#### 示例

```sh
python MetDetPy.py ./test/20220413Red.mp4 --mask ./test/mask-east.jpg
```

#### 自定义配置文件

大多数与检测相关的重要参数都预先定义并储存在配置文件中。大多数情况下，这些预设值都能够较好的工作，但有时对参数微调可以取得更好的结果。如果想要获取配置文件的说明，可以参考[配置文件文档](./config-doc-cn.md)获取更多信息。

### 运行图像流星检测

要启动图像流星检测器，运行 `run_model.py`：

```sh
python run_model.py target 
```

(相关能力还在建设中)

### 其他工具的使用

#### ClipToolkit

ClipToolkit可用于一次性创建一个视频中的多段视频切片或这些视频段的堆栈图像。其用法如下：

```sh
python ClipToolkit.py target json [--mode {image,video}] [--suffix SUFFIX] [--save-path SAVE_PATH] [--resize RESIZE] [--jpg-quality JPG_QUALITY] [--png-compressing PNG_COMPRESSING]
```
##### Arguments:

* target: 目标视频文件。

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

注意：如果使用 JSON 格式的字符串而不是 JSON 文件的路径，需要注意命令行中双引号的转义。

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

* --debug：用这个启动`evaluate.py`时，会有详细的调试信息。

## 打包Python代码为可执行程序

我们提供了 [make_package.py](../make_package.py) 来将MetDetPy打包为独立的可执行程序。该工具支持使用 `pyinstaller` 或 `nuitka` 来打包/编译。

当使用该脚本时，请确保至少安装了`pyinstaller` 或 `nuitka` 中的任意一个工具。此外，在使用 `nuitka` 作为编译工具时，请确保在您的设备上有至少一个C/C++编译器可用。

该工具的用法如下：

```sh
python make_package.py [--tool {nuitka,pyinstaller}] [--mingw64]
     [--apply-upx] [--apply-zip] [--version VERSION]
```

* --tool: 使用的打包/编译工具。应当从 {nuitka,pyinstaller} 中选择。默认的编译器为 `nuitka` 。

* --mingw64: 使用MinGW64作为编译器。该选项仅在Windows上使用 `nuitka` 进行编译时生效。

* --apply-upx: 启用UPX以压缩可执行程序的大小。仅当使用 `nuitka` 进行编译时生效。

* --apply-zip: 打包完成时同时生成Zip压缩包。

* --version: 指定 MetDetPy 的版本号（仅用于文件名中）。当空缺时默认使用 `./MetLib/utils.py` 中的版本号。

目标可执行程序的目录会生成在 [dist](../dist/) 目录下。

注意：

1. 建议使用 `Python>=3.9`, 且安装 `pyinstaller>=5.0` 或 `nuitka>=1.3.0` 以避免兼容性问题。任一工具都可以用于创建可执行程序。

3. 由于Python的特性，上述两种工具均无法跨平台打包生成可执行文件。

4. （原因不明）如果环境中有`matplotlib`或`scipy`，它们很可能会一起打包到最终目录中。 为避免这种情况，建议使用干净的环境进行打包。

### 性能和效率

1. 在 3840x2160 10fps 视频上应用默认配置进行检测时，MetDetPy 检测流星的平均时间开销为视频长度的 20-30%（使用 Intel i5-7500 测试）。 FPS 较高的视频可能会花费更多时间。

2. 我们使用从各种设备（从改装监控摄像头到数码相机）拍摄的样本视频测试 MetDetPy，MetDetPy 平均能够达到 80% 以上的准确率和 80% 以上的召回率。

3. MetDetPy 现在可以快速高效地检测大多数流星视频。 但当面对复杂的天气或其他影响因素时，其准确率和召回率还有待提高。 如果您发现 MetDetPy 在您的视频上表现不够好，欢迎联系我们或提交问题（如果可以的话，一并提供完整或剪辑的视频）。

## 许可

该项目根据 Mozilla 公共许可证 2.0 (MPL-2.0) 获得许可。这意味着您可以自由使用、修改和分发该软件，但须满足以下条件：

* 源代码可用性：您对源代码所做的任何修改也必须在 MPL-2.0 许可证下可用。这确保了社区可以从改进和变化中受益。
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

见 [更新日志](update-log.md)。

### 统计数据

[![Star History Chart](https://api.star-history.com/svg?repos=LilacMeteorObservatory/MetDetPy&type=Timeline)](https://star-history.com/#LilacMeteorObservatory/MetDetPy&Timeline)