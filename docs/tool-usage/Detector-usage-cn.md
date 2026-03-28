# 检测工具使用指南

<center> 语言：<a href="./Detector-usage.md">English</a> | <b>简体中文</b> </center>

MetDetPy 项目提供了两种检测工具：`MetDetPy` 用于视频流星检测，`MetDetPhoto` 用于图像流星检测。通过与`ClipToolkit` 结合使用，可以实现视频流星的检测-导出工作流。

## 目录

* [快速开始](#快速开始)
* [工具对比](#工具对比)
* [MetDetPy - 视频流星检测工具](#metdetpy---视频流星检测工具)
    * [完整参数说明](#metdetpy-完整参数说明)
    * [使用示例](#metdetpy-使用示例)
* [MetDetPhoto - 图像流星检测工具](#metdetphoto---图像流星检测工具)
    * [完整参数说明](#metdetphoto-完整参数说明)
    * [使用示例](#metdetphoto-使用示例)
* [常见问题](#常见问题)

---

## 快速开始

MetDetPy 与 MetDetPhoto 设计有合理的默认配置，在多数情况下无需额外参数即可直接运行。

### 工具选用

- 如果是**直录流星的视频文件**（fps>=4），使用 `MetDetPy`;
- 如果您是**静态图像**、**图像文件夹（如延时序列图像）**或**延时视频**，使用 `MetDetPhoto`。

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

# 检测后,过滤掉所有的负样本和小尺寸结果，拷贝正样本到另一个文件夹
python ClipToolkit.py results.json --enable-filter-rules  --save-path ./output

# 或导出带标注框的图像
python ClipToolkit.py results.json --enable-filter-rules --with-bbox --save-path ./output
```


## MetDetPy - 视频流星检测工具

`MetDetPy` 是 MetDetPy 项目的视频流星检测启动器，可用于从视频文件中检测流星事件。

### MetDetPy 完整参数说明

```sh
python MetDetPy.py target [--cfg CFG] [--mask MASK] [--start-time START_TIME] [--end-time END_TIME]
               [--exp-time EXP_TIME] [--mode {backend,frontend}] [--debug]
               [--resize RESIZE] [--adaptive-thre ADAPTIVE_THRE] [--bi-thre BI_THRE | --sensitivity SENSITIVITY]
               [--recheck RECHECK] [--save-rechecked-img SAVE_RECHECKED_IMG]
               [--provider {cpu,default,coreml,dml,cuda}] [--live-mode {on,off}] [--save-path SAVE-PATH]
               [--resource-dir RESOURCE_DIR]
```

#### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `target` | 待检测视频文件路径。支持 H264、HEVC 等常见视频编码 | 必需 |
| `--cfg` | 配置文件路径 | `./config/m3det_normal.json` |
| `--mask` | 掩模（遮罩）图像路径 | 无 |
| `--start-time` | 检测开始时间（毫秒数或 `"HH:MM:SS"` 格式） | 0（视频开始） |
| `--end-time` | 检测结束时间（毫秒数或 `"HH:MM:SS"` 格式） | 视频结束 |
| `--mode` | 运行模式：`frontend`（显示进度条）或 `backend`（管道模式） | `frontend` |
| `--debug` | 启用调试模式，打印详细日志 | 关闭 |
| `--visu` | 启用可视化窗口，实时显示检测过程 | 关闭 |
| `--live-mode` | 实时模式：检测时间接近实际视频时长，均衡 CPU 开销 | `off` |
| `--provider` | 模型推理后端，可选 `cpu`、`default`、`coreml`、`dml`、`cuda` | `default` |
| `--save-path` | 保存检测结果到 MDRF 格式 JSON 文件的路径 | 不保存 |

#### 额外参数（可覆盖配置文件）

以下参数在不设置时使用配置文件中的默认值。有关配置文件的详细说明，请参考[配置文件文档](../config-doc-cn.md)。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--resize` | 检测时使用的帧图像尺寸。可指定为整数（如 `960`，长边）、列表（如 `[960,540]`）或字符串（如 `960x540`） | 配置文件中的值 |
| `--exp-time` | 单帧曝光时间。可指定浮点数或从 `{auto, real-time, slow}` 中选择 | `auto` |
| `--adaptive-thre` | 是否启用自适应二值化阈值。从 `{on, off}` 中选择 | 配置文件中的值 |
| `--bi-thre` | 二值化阈值。当启用自适应二值化时该选项无效。不能与 `--sensitivity` 同时使用 | 配置文件中的值 |
| `--sensitivity` | 检测器灵敏度。从 `{low, normal, high}` 中选择。不能与 `--bi-thre` 同时使用 | 配置文件中的值 |
| `--recheck` | 是否启用重校验机制减少误报。从 `{on, off}` 中选择 | 配置文件中的值 |
| `--save-rechecked-img` | 保存重校验后的图像的路径 | 不保存 |

### MetDetPy 使用示例

一个常见的组合参数示例如下（通常不需要全部启用）：

```sh
python MetDetPy.py video.mp4 \
    --mask mask.jpg \ # 使用掩模排除干扰区域
    --save-path result.json \ # 保存结果到 JSON 文件
    --start-time 00:01:00 --end-time 00:05:00 \ # 从 00:01:00 开始检测到 00:05:00
    --visu \ # 启用可视化窗口
    --debug \ # 启用详细调试信息
    --live-mode \ # 实时模式
    --resize 960x540 \ # 检测时使用的帧图像尺寸
    --exp-time 1000 \ # 单帧曝光时间
    --adaptive-thre on \ # 启用自适应二值化阈值
    --sensitivity normal \ # 检测器灵敏度
```

---

## MetDetPhoto - 图像流星检测工具

`MetDetPhoto` 是 MetDetPy 项目的图像流星检测启动器，可用于从单张图像、图像文件夹或延时视频中检测流星。

### MetDetPhoto 完整参数说明

```sh
python MetDetPhoto.py target [--mask MASK]
                             [--model-path MODEL_PATH] [--model-type MODEL_TYPE]
                             [--exclude-noise] [--debayer] [--debayer-pattern DEBAYER_PATTERN]
                             [--visu] [--visu-resolution VISU_RESOLUTION]
                             [--save-path SAVE_PATH]
                             [--resource-dir RESOURCE_DIR]
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `target` | 目标图像文件/文件夹。支持单张图像、图像文件夹以及常规视频编码格式的延时视频文件 | 必需 |
| `--mask` | 掩模（遮罩）图像路径 | 无 |
| `--model-path` | 模型权重文件路径 | `./weights/yolov5s_v2.onnx` |
| `--model-type` | 模型格式，决定如何处理模型的输出 | `YOLO` |
| `--exclude-noise` | 排除常见噪声类型（如卫星和飞虫），仅保存正样本到文件 | 关闭 |
| `--debayer` | 在处理延时视频前对视频帧进行 Debayer 变换 | 关闭 |
| `--debayer-pattern` | Debayer 使用的矩阵，如 `RGGB` 或 `BGGR`。仅在 `--debayer` 选项启用时生效 | 无 |
| `--visu` | 启用可视化窗口，显示当前的检测情况 | 关闭 |
| `--visu-resolution` | 可视化窗口的分辨率设置 | 默认分辨率 |
| `--save-path` | 保存检测结果到 [MDRF](../data-format-cn.md#流星检测记录格式(MDRF)) 格式文件中 | 不保存 |

### MetDetPhoto 使用示例

#### 基础用法

```sh
# 检测单张图像
python MetDetPhoto.py image.jpg

# 检测图像文件夹中的所有图像
python MetDetPhoto.py ./images

# 检测延时视频中的流星帧，保存检测结果
python MetDetPhoto.py timelapse.mp4  --save-path result.json
```

#### MetDetPhoto 使用示例

一个常见的组合参数示例如下（通常不需要全部启用）：

```sh
python MetDetPhoto.py ./images \
    --mask mask.jpg # 使用掩模排除干扰区域 \
    --model-path ./custom_model.onnx # 使用自定义模型权重文件 \
    --exclude-noise # 仅保存流星检测结果，排除卫星、飞虫等常见噪声 \
    --debayer --debayer-pattern RGGB # 处理 RAW 格式延时视频 \
    --visu  --visu-resolution 1920x1080 # 启用可视化窗口 \
    --save-path results.json # 保存检测结果到 JSON 文件
```

---

## 输出说明

### 命令行输出

两个检测工具都会在运行时实时输出检测信息到命令行，包括：

- 检测进度
- 检测到的流星样本信息
- 运行时间统计

### 结果文件

使用 `--save-path` 参数时，检测结果会被保存为 [MDRF](../data-format-cn.md#流星检测记录格式(MDRF)) 格式的 JSON 文件。该文件包含：

- 视频/图像信息（文件路径、分辨率、帧率等）
- 检测结果的详细信息
- 每个检测的时间戳、边界框坐标、置信度等

该结果文件可以被 `ClipToolkit` 等工具处理，用于导出流星片段、生成堆栈图像等。

---

## 常见问题

### Q1: 如何创建掩模图像？

**A:** 掩模图像用于排除不需要检测的区域（如建筑物、树木等）。创建方法：

1. 创建一张空白图像（尺寸不限，会自动缩放到视频/图像尺寸）
2. 使用任何非白色颜色在需要排除的区域上涂抹（推荐黑色或者红色）
3. 保存为 JPEG 或 PNG 格式

### Q2: 检测结果不理想怎么办？

**A:** 可以尝试以下方法：

**对于 MetDetPy：**
- 调整使用的配置文件
- 使用掩模图像排除干扰区域

**对于 MetDetPhoto：**
- 使用掩模排除干扰区域
- 启用 `--exclude-noise` 过滤噪声

### Q3: 如何导出检测到的流星片段？

**A:** 使用 `ClipToolkit` 工具：

```sh
# 1. 先运行检测并保存结果
python MetDetPy.py video.mp4 --save-path results.json

# 2. 使用 ClipToolkit 导出片段
python ClipToolkit.py results.json --mode video --save-path ./output
```

详见 [ClipToolkit 使用指南](./ClipToolkit-usage-cn.md)。

### Q4: `--live-mode` 有什么作用？

**A:** `--live-mode` 会将检测速度控制在接近实际视频时长，适用于实时监控场景，可以帮助均衡 CPU 开销。仅适用于 `MetDetPy`。

### Q7: 何时需要使用 `--debayer`？

**A:** 当处理的延时视频是 RAW 格式（如使用专业相机拍摄的 RAW 视频）时，需要使用 `--debayer` 参数进行 Debayer 变换。需要根据相机的传感器设置指定正确的 Bayer 模式（如 `RGGB`、`BGGR` 等）。

---

## 相关文档

- [ClipToolkit 使用指南](./ClipToolkit-usage-cn.md)
- [配置文件说明](../config-doc-cn.md)
- [数据格式说明](../data-format-cn.md)
- [流星检测记录格式 (MDRF)](../data-format-cn.md#流星检测记录格式(MDRF))
