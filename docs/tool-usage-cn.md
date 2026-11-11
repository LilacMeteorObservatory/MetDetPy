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
     [--apply-upx] [--apply-zip] [--onefile] [--clean-dist]
     [--build-tag TAG] [--build-report]
     [--mingw64] [--macos-sign-identity IDENTITY]
     [--windowed] [--icon ICON_PATH]
```

* `--backend`：选择 `nuitka` 或 `pyinstaller`，默认为 `nuitka`。

* `--apply-upx`：启用 UPX 以压缩可执行程序。

* `--apply-zip`：打包完成后生成 ZIP 压缩包。

* `--onefile`：每个程序生成一个可执行文件，而不是目录式程序包。

* `--clean-dist`：打包前先删除 `dist/` 目录，避免旧的可执行文件和资源目录混入新构建。Nuitka 与 PyInstaller 后端均生效。

* `--build-tag`：可选，覆盖 Nuitka onefile 固定解压目录的缓存标识。缓存标识默认为 `VERSION`（例如 `V2.6.1`），每个版本自然使用新的解压目录；只有同版本多次构建需要互相隔离时才需要指定，修改后必须重新打包才会生效。

* `--build-report`：可选，在 `make_package.py` 同目录写出 `build_report.txt`，只记录版本、源码提交/工作区状态、工具链版本和各产物 SHA256。

* `--mingw64`：Nuitka 后端在 Windows、Python 3.12 及更早版本使用 MinGW64；Python 3.13+ 自动改用 MSVC，需要安装 Visual Studio Build Tools 的 C++ 桌面开发组件。

* `--macos-sign-identity`：Nuitka 后端使用的 macOS 签名身份。

* `--windowed`：PyInstaller 后端使用无控制台窗口模式。

* `--icon`：PyInstaller 后端使用的可执行文件图标。

### Nuitka onefile 固定解压目录

使用 `--backend nuitka --onefile` 时，生成的可执行文件使用固定的解压缓存目录，而不是每次启动都创建新的临时目录：

```
{CACHE_DIR}/MetDetPy/<VERSION>
```

`{CACHE_DIR}` 由 Nuitka 在运行时解析（Windows 为 `%LOCALAPPDATA%`；macOS / Linux 为 `$XDG_CACHE_HOME` 或 `~/.cache`）。首次启动会把载荷解压到该目录，后续启动直接复用已解压的文件（Nuitka 会校验每个文件的 CRC32），从而省去每次启动的解压开销。缓存会在退出后保留，因此仍会占用解压后的磁盘空间。同一构建的三个 onefile 程序共用同一个缓存目录，因为它们的公共依赖相同。

缓存以 `VERSION` 为标识，因此新版本不会复用旧版本的载荷。同版本重新构建会复用同一个缓存目录（Nuitka 会校验每个文件的 CRC32 并从载荷刷新变化的文件）；确需让同版本的多次构建互不干扰时，才使用 `--build-tag`。旧的 `onefile_*` 临时目录和旧版本缓存不会自动清理，可在确认相关程序全部退出后手动删除。

onefile 模式下会显式处理 `pyexiv2` 原生组件：构建时注册 `exiv2api` 扩展，并把 `exiv2.dll` / `libexiv2` 共享库放入载荷内的 `pyexiv2/lib`，以保证导出图片时 EXIF/ICC 信息仍能正常写入。

**Windows 商店版（MSIX / APPX）必须打引导层补丁**：Windows 文件系统虚拟化会把 onefile 解压目录（`%LOCALAPPDATA%\MetDetPy\<VERSION>`）重定向到包私有目录，而 Nuitka 引导层随后用逻辑路径启动内部子进程 → 加载器找不到同目录的 `python310.dll` → `0xC0000135` 秒退。Windows + onefile 构建前会自动执行 `nuitka-patches/patch_onefile_bootstrap.py`（锁定 Nuitka 4.0.5），把逻辑路径解析为物理路径后再 `CreateProcessW`；版本或锚点不匹配、或产物里找不到补丁标记时**构建直接失败**，不会静默产出商店版起不来的包。补丁会改写构建环境 `site-packages` 里的 `OnefileBootstrap.c`，原始文件备份为 `.metdetpy-orig`。商店版验收必须在新账号 / 隔离机上，且普通缓存与包私有缓存都不存在的条件下首启验证（例如新版本号，或显式指定新的 `--build-tag`）。

Windows onefile 发行版示例：

```powershell
python make_package.py --backend nuitka --onefile --apply-zip --clean-dist `
  --build-report
```

该命令会生成 `dist/MetDetPy.exe`、`dist/MetDetPhoto.exe`、`dist/ClipToolkit.exe`，`config/`、`weights/`、`global/` 资源目录，以及 `dist/MetDetPy-win-v2.6.1-nuitka-onefile.zip`。指定 `--build-report` 时，会在 `make_package.py` 同目录写出精简的 `build_report.txt`，只记录版本、源码提交/工作区状态、工具链版本和各产物 SHA256。

### 通用说明

可执行程序和可选的 ZIP 压缩包会生成在 [dist](../dist/) 目录下。

使用 Nuitka 后端时，ZIP 命名格式为
`MetDetPy-<平台>-<版本>-nuitka-<onefile|onedir>.zip`。onefile 包的根目录直接是
可执行文件和资源目录；onedir 包内是 `MetDetPy/` 应用目录，其中包含三个可执行文件、
运行时依赖以及 `config/`、`weights/`、`global/` 资源目录。

注意：

1. 使用 Nuitka 时需要安装 `nuitka>=2.0.0` 并准备可用的 C/C++ 编译器；使用 PyInstaller 时请安装 `pyinstaller>=6.0`。
2. 由于Python的特性，这些工具均无法跨平台打包生成可执行文件。你只能打包当前平台的可执行程序。
3. PyInstaller 默认排除绘图、训练、交互式开发及 Qt 等可选依赖，只打包三个运行工具需要的依赖；开发用照片差异统计图工具不包含在发行包中。建议仍使用独立虚拟环境构建，并保留 `pyexiv2` 等原生库。

建议在独立环境中按运行依赖清单安装，避免开发环境的可选导入进入发行包。例如 Windows：

```powershell
python -m venv .venv-package
.venv-package\Scripts\python.exe -m pip install -r requirements.txt pyinstaller
.venv-package\Scripts\python.exe make_package.py --backend pyinstaller
```

这份清单控制构建环境的直接依赖，其必需的间接依赖由 pip 安装，再由 PyInstaller 分析收集；不是对最终模块表做截断。Windows DML 发行包应在构建环境中使用 `onnxruntime-directml` 替代通用 `onnxruntime`。
