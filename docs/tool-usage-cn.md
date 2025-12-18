# 工具用法

<center> 语言：<a href="./tool-usage.md">English</a> | <b>简体中文</b> </center>

MetDetPy提供了一些用于支持相关功能的工具。

## Menu

* [ClipToolkit - (批)图像堆栈或视频切片工具](#cliptoolkit)
* [Evaluate - 性能评估，效果测试工具](#evaluate)
* [make_package - 打包可执行程序工具](#打包Python代码为可执行程序)

## ClipToolkit

`ClipToolkit`（切片工具）可用于一次性创建一个视频中的多段视频切片或这些视频段的堆栈图像。从`MetDetPy v2.2.0`开始，ClipToolkit扩展了调用方式，以支持更灵活的使用和更通用的场景。

ClipToolkit的完整参数列表如下：

```sh
python ClipToolkit.py target [json] [--start-time START_TIME] [--end-time END_TIME]
                      [--mode {image,video}] [--suffix SUFFIX]
                      [--save-path SAVE_PATH] [--resize RESIZE]
                      [--jpg-quality JPG_QUALITY][--png-compressing PNG_COMPRESSING]
                      [--debayer] [--debayer-pattern DEBAYER_PATTERN]
                      [--with-annotation][--with-bbox]
                      [--debug]
                      
```

### 用法简介

ClipToolkit 共可接受3种输入模式，这主要通过传入的位置参数(Positional Arguments)决定。这些模式和对应的位置参数用法如下：

1. 通用模式
    通用模式是从 `v1.3.0` 开始的基本模式，可用于一次性创建多个叠加图或视频切片,并配置每一个图/视频的名称与路径。要使 ClipToolkit 在通用模式下运行，需要传入两个位置参数：`target`（目标视频文件路径）与 `json` （JSON格式的字符串或者JSON文件的路径）。
    其中，json 应该是一个包含了若干组起始时间，结束时间和文件名（可选）信息的json数组，每个元素都应该包含：

    * 必要`"time"`键，其值应是两个`"hh:mm:ss.ms"`格式的字符串组成的数组，表示片段的开始时间和结束时间；
    * 可选的`"filename"` 键，您可以在其值中指定文件名和后缀（即视频剪辑应该转换为何种格式并命名。）
    * 可选的 `"target"` 键，需要为数组，每一项包含`"pt1"`,`"pt2"`和`"preds"`，表明标注框位置和类别。

    此模式下，优先使用`JSON`中指定的文件名和类型选项。可选参数将仅在无对应参数时生效。JSON的一个实例为 [clip_test.json](../test/clip_test.json)。使用例如下：

```sh
python ClipToolkit.py ./test/20220413Red.mp4 ./test/clip_test.json --mode image --suffix jpg --jpg-quality 60 --resize 960x540 --save-path ./test
```

2. 精简模式
    精简模式通过仅传入一个视频文件作为位置参数以启用，用于产生单个图像堆栈或视频切片。产生文件的起止时间，格式和其他参数需要通过可选参数指定，默认为完整视频堆栈一张`JPG`格式的图像作为结果。这种方式下不需要构建`JSON`文件或转义字符串，更容易在命令行场景使用。使用例如下：

```sh
python ClipToolkit.py ./test/20220413Red.mp4 --start-time 00:03:00 --end-time 00:05:00 --mode image --output-name ./test/generated_img.jpg
```

3. 样本生成模式
    当希望为检测结果（或标注）生成所有对应的叠加图或视频切片时，可以通过仅指定一个`MDRF`格式的`JSON`文件作为输入（有关该文件格式的介绍可见[流星检测记录格式 (MDRF)](#meteor-detection-recording-format-mdrf)。该格式文件可从`MetDetPy v2.2.0`之后的`evaluate`或`MetDetPy`生成）来启用该模式。其余参数通过可选参数指定。这可以用于生成批量的标注图像，用于微调模型。使用例如下：

```sh
python ClipToolkit.py ./test/video_mdrf.json --mode video
``` 

### 可选参数

ClipToolkit支持的可选参数列表如下：

* `--start-time`: 片段起始时间。可使用毫秒数或 `HH:MM:SS.ms` 格式指定。仅在简单模式下有效。默认为视频起始时间。

* `--end-time`: 片段结束时间。可使用毫秒数或 `HH:MM:SS.ms` 格式指定。仅在简单模式下有效。默认为视频结束时间。

* `--mode`：将剪辑转换为图像或视频。 应从 `{image, video}` 中选择。 此选项会被 json 中的特定文件名覆盖。

* `--suffix`：输出的后缀。 默认情况下，图像模式为“jpg”，视频模式为“avi”。 此选项可以被 JSON 中的特定文件名覆盖。

* `--save-path`：放置图像/视频的路径。 当 JSON 中只包含一个片段时，您可以在 `--save-path` 中包含文件名以简化您的 JSON。

* `--resize`：将图像/视频调整为给定的分辨率。

* `--with-annotation`: 同时输出 labelme 风格的标注json文件（仅支持通用模式和样本生成模式，需要附带有目标标注）。

* `--with-bbox`: 在导出的图像中绘制检测框（仅支持通用模式和样本生成模式，需要附带有目标标注）。

* `--png-compressing`: 生成的png图像压缩程度。 其值应为$Z \in [0,9]$； 默认情况下取值为3。

* `--jpg-quality`: 生成的jpg图像的质量。 其值应为$Z \in [0,100]$； 默认情况下取值为95。

* `--debug`: 是否在debug模式下运行以打印更详细的信息

注意：如果使用 JSON 格式的字符串而不是 JSON 文件的路径，需要注意命令行中双引号的转义。


## Evaluate

Evaluate 是一个集成了性能评估及效果测试工具。它可以用于生成运行结果报告，评估对设备资源的占用，比较结果间的差异。

若需要评估MetDetPy在某个视频上的检测性能，可以运行 `evaluate.py` :

```sh
python evaluate.py json [--cfg CFG] [--load LOAD] [--save SAVE] [--metric] [--debug]
```

### 参数

* `json`：一个`MDRF`格式的JSON文件，里面需要至少包含视频相关的必要信息（视频文件和掩模文件路径，起止时间）以启动。它的格式应该是满足[流星检测记录格式 (MDRF)](#meteor-detection-recording-format-mdrf)中的要求。

* `--cfg`：配置文件。 默认使用默认配置，即[m3det_normal.json](../config/m3det_normal.json)。

* `--load`: 如果启用并填写了另一个`JSON`的路径，`evaluate.py` 将直接加载其结果作为本次检测结果进行比较，而不运行检测。

* `--save`：要将检测结果保存到的路径与文件名。

* `--metrics`：依据提供json文件的类别不同，进行回归测试（与其他预测结果比较）或计算检测的精度和召回率（与基本事实比较）。 要应用该选项，`json` 文件中需要包含`results` 的信息。

* `--debug`：用这个启动`evaluate.py`时，会有详细的调试信息。

### Example
(To be updated)

## 打包Python代码为可执行程序

我们提供了 [make_package.py](../make_package.py) 来将MetDetPy打包为独立的可执行程序。该工具使用 `nuitka` 来打包/编译。

当使用该脚本时，请确保安装了 `nuitka` ，并确保在您的设备上有至少一个C/C++编译器可用。

该工具的用法如下：

```sh
python make_package.py [--tool {nuitka}] [--mingw64]
     [--apply-upx] [--apply-zip] [--version VERSION]
```

* `--tool`: 使用的打包/编译工具。应当从 {nuitka,pyinstaller} 中选择。默认的编译器为 `nuitka` 。

* `--mingw64`: 使用MinGW64作为编译器。该选项仅在Windows上使用 `nuitka` 进行编译时生效。

* `--apply-upx`: 启用UPX以压缩可执行程序的大小。仅当使用 `nuitka` 进行编译时生效。

* `--apply-zip`: 打包完成时同时生成Zip压缩包。

* `--version`: 指定 MetDetPy 的版本号（仅用于文件名中）。当空缺时默认使用 `./MetLib/utils.py` 中的版本号。

目标可执行程序的目录会生成在 [dist](../dist/) 目录下。

注意：

1. 建议使用 `Python>=3.9`, 且安装 `nuitka>=2.0.0` 以避免兼容性问题。

2. 由于Python的特性，`nuitka` 均无法跨平台打包生成可执行文件。你只能打包当前平台的可执行程序。