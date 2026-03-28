# 工具用法

<center> 语言：<a href="./tool-usage.md">English</a> | <b>简体中文</b> </center>

MetDetPy提供了一些用于支持相关功能的工具。

## 目录

* [工具使用指南](#工具使用指南)
    * [检测工具 - MetDetPy 和 MetDetPhoto](#检测工具)
    * [ClipToolkit - (批)图像堆栈或视频切片工具](#cliptoolkit)
* [其他工具](#其他工具)
    * [Evaluate - 性能评估，效果测试工具](#evaluate)
    * [make_package - 打包可执行程序工具](#make-package)

## 工具使用指南

### 检测工具

MetDetPy 提供了两种检测工具：`MetDetPy` 用于视频流星检测，`MetDetPhoto` 用于图像流星检测。这两个工具各有特点，适用于不同的使用场景。

要了解如何使用这些检测工具，请参考 [检测工具使用指南](./tool-usage/Detector-usage-cn.md)。

---

### ClipToolkit

`ClipToolkit`（切片工具）可用于一次性创建一个视频中的多段视频切片或这些视频段的堆栈图像。要了解如何使用该工具，请参考 [ClipToolkit 使用指南](./tool-usage/ClipToolkit-usage-cn.md)。

---

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

---

## make_package

我们提供了 [make_package.py](../make_package.py) 来将MetDetPy打包为独立的可执行程序。该工具使用 `nuitka` 来打包/编译。

当使用该脚本时，请确保安装了 `nuitka` ，并确保在您的设备上有至少一个C/C++编译器可用。

该工具的用法如下：

```sh
python make_package.py [--tool {nuitka}] [--mingw64]
     [--apply-upx] [--apply-zip] [--version VERSION]
     [--onefile]
```

* `--tool`: 使用的打包/编译工具。应当从 {nuitka,pyinstaller} 中选择。默认的编译器为 `nuitka` 。

* `--mingw64`: 使用MinGW64作为编译器。该选项仅在Windows上使用 `nuitka` 进行编译时生效。

* `--apply-upx`: 启用UPX以压缩可执行程序的大小。仅当使用 `nuitka` 进行编译时生效。

* `--apply-zip`: 打包完成时同时生成Zip压缩包。

* `--version`: 指定 MetDetPy 的版本号（仅用于文件名中）。当空缺时默认使用 `./MetLib/utils.py` 中的版本号。

* `--onefile`: 生成单文件可执行程序（onefile模式）。使用此模式时，需要确保静态资源文件夹（`config/`、`weights/`、`resource/`、`global/`）放置在可执行文件旁边，或在运行时使用 `--resource-dir` / `-R` 选项指定其位置。

目标可执行程序的目录会生成在 [dist](../dist/) 目录下。

注意：

1. 建议使用 `Python>=3.9`, 且安装 `nuitka>=2.0.0` 以避免兼容性问题。

2. 由于Python的特性，`nuitka` 均无法跨平台打包生成可执行文件。你只能打包当前平台的可执行程序。
