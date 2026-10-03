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
python evaluate.py --report REPORT [--cfg CFG] [--load LOAD] [--save-path SAVE_PATH] [--metric] [--debug]
```

### 参数

* `json`：一个`MDRF`格式的JSON文件，里面需要至少包含视频相关的必要信息（视频文件和掩模文件路径，起止时间）以启动。它的格式应该是满足[流星检测记录格式 (MDRF)](#meteor-detection-recording-format-mdrf)中的要求。

* `--cfg`：配置文件。 默认使用默认配置，即[m3det_normal.json](../config/m3det_normal.json)。

* `--load`: 如果启用并填写了另一个`JSON`的路径，`evaluate.py` 将直接加载其结果作为本次检测结果进行比较，而不运行检测。

* `--save`：要将检测结果保存到的路径与文件名。

* `--metric`：与参考结果比较，并计算分类 Precision / Recall / F1。`json` 文件中需要包含 `results`。

* `--metrics-path`：与 `--metric` 一起使用，将每类指标、micro/macro 汇总和 P-R 数据点保存到 JSON。

报告的 `comparison_summary` 从原有比较矩阵提炼：`added_num` 为未匹配的新预测数，`lost_num` 为未匹配的参考目标数，`category_changed_num` 为时空匹配但类别不同的目标数；`category_transitions` 按参考类别 `from_category` → 新类别 `to_category` 列出计数。类别变化不重复计入新增或丢失。该摘要沿用原比较口径：与预测基线比较时包含低分和 DROPPED 候选；与 GT 比较时分数不超过 0.5 的新预测不参与匹配。它与有效类别 P/R/F1 的过滤口径不同。

* `--batch`：输入 JSON 为 case 清单；`--case ID` 可重复指定，只执行选中的 case。
* `--passes N`：每个 case 重复运行次数，默认 1；单报告也支持重复运行。
* `--output-dir`：批量或重复运行的输出目录，默认 `evaluation-results`。每次运行独立进程，分别保存预测、指标、差异和日志，`summary.json` 保存逐 case/pass 结果及成功运行的性能中位数、最小值、最大值。失败记录后继续执行，最终退出码为 1。重复运行不是独立效果样本，不能把各 pass 的 TP/FP/FN 累加作为数据集质量指标。输出目录再次使用会覆盖同编号的运行文件，比较版本时应使用不同目录。

清单示例（`json`/可选 `cfg` 相对于清单目录；省略 `cfg` 时使用命令行配置）：

```json
{"cases": [
  {"id": "night", "json": "night.json"},
  {"id": "noise", "json": "noise.json", "cfg": "configs/noise.json"}
]}
```

```sh
python evaluate.py --manifest cases.json --metric --output-dir results/base
python evaluate.py --manifest cases.json --case night --passes 3 --output-dir results/night
```

性能统计使用 `perf_counter` 测量检测调用的总耗时（包含初始化与收尾，不含采样线程退出等待）。`cpu_time` 为进程 user+system 时间增量，`avg_cpu_usage` 为该增量除以总耗时乘 100；100% 表示占用一个逻辑核心，可超过 100%。内存单位 MiB，保留平均 RSS，并新增采样峰值、起止值及净增长、采样次数和间隔。默认每 0.5 秒采样，峰值可能遗漏短暂尖峰；平均值是样本平均。统计限当前进程，不包含外部子进程或 GPU 显存。

分类指标使用默认分数阈值 0.5（严格大于），先按置信度降序进行类别无关的一对一时空匹配，选择交叠乘积最大的未匹配参考目标。错分类分别计入预测类别 FP 和真实类别 FN；`matched_num` 单独表示时空匹配数量。`DROPPED` 和 GT 中分数不超过阈值的标注不参与分类指标，但预测总数、低分候选数及 DROPPED 数量单独保留。零分母指标返回 0；macro 只汇总在当前阈值下出现的类别。

P-R 数据点按所有有效候选的不同分数扫描阈值，包含全排除与全保留端点；仅反映报告中保留的候选，不恢复上游已过滤或已转成 DROPPED 的目标。与预测基线比较时，指标表示一致性，不代表真实检测准确率。

* `--debug`：用这个启动`evaluate.py`时，会有详细的调试信息。

### Example
```sh
python evaluate.py --report annotation.json --load predictions.json --metric --metrics-path metrics.json
```

---

## make_package

使用 [make_package.py](../make_package.py) 可以通过 Nuitka 或 PyInstaller
将 MetDetPy 打包为独立可执行程序。默认使用 Nuitka 后端。

```sh
python make_package.py [--backend {nuitka,pyinstaller}]
     [--apply-upx] [--apply-zip] [--onefile]
     [--mingw64] [--macos-sign-identity IDENTITY]
     [--windowed] [--icon ICON_PATH]
```

* `--backend`：选择 `nuitka` 或 `pyinstaller`，默认为 `nuitka`。

* `--apply-upx`：启用 UPX 以压缩可执行程序。

* `--apply-zip`：打包完成后生成 ZIP 压缩包。

* `--onefile`：每个程序生成一个可执行文件，而不是目录式程序包。

* `--mingw64`：Nuitka 后端在 Windows 上使用 MinGW64 编译器。

* `--macos-sign-identity`：Nuitka 后端使用的 macOS 签名身份。

* `--windowed`：PyInstaller 后端使用无控制台窗口模式。

* `--icon`：PyInstaller 后端使用的可执行文件图标。

### 通用说明

可执行程序和可选的 ZIP 压缩包会生成在 [dist](../dist/) 目录下。

注意：

1. 使用 Nuitka 时需要安装 `nuitka>=2.0.0` 并准备可用的 C/C++ 编译器；使用 PyInstaller 时请安装 `pyinstaller>=6.0`。
2. 由于Python的特性，这些工具均无法跨平台打包生成可执行文件。你只能打包当前平台的可执行程序。
3. 如果你的环境中存在 `matplotlib` 或 `scipy`，它们可能会被打包进去。如果想要减小打包体积，请准备一个干净的环境或避免安装这些重量级依赖。
