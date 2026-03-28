# ClipToolkit 使用指南

<center> 语言：<a href="./ClipToolkit-usage.md">English</a> | <b>简体中文</b> </center>

`ClipToolkit`（切片工具）可用于一次性创建一个视频中的多段视频切片或这些视频段的堆栈图像。从 `MetDetPy v2.2.0` 开始，ClipToolkit 扩展了调用方式，以支持更灵活的使用和更通用的场景。

## 目录

* [快速开始](#快速开始)
* [三种输入模式](#三种输入模式)
    * [通用模式](#通用模式)
    * [精简模式](#精简模式)
    * [样本生成模式](#样本生成模式)
* [完整参数说明](#完整参数说明)
* [导出格式支持](#导出格式支持)
* [高级功能](#高级功能)
    * [去噪算法](#去噪算法)
    * [Labelme 标注导出](#labelme-标注导出)
    * [过滤规则](#过滤规则)
* [配置文件](#配置文件)
* [常见问题](#常见问题)

---

## 快速开始

> **提示**：本文档中的 `[video.mp4]`、`[detection_result.json]` 等表示需要替换为实际文件路径的占位符。

### 快速导出切片

ClipToolkit 支持快速导出一个指定起止时间的视频片段或是最大值堆栈图像：

```sh
# 从视频中截取一张堆栈图像
# 不指定 --save-path 则默认保存到当前目录，并使用"视频名称_起始时间-结束时间"作为文件名。如下面的输入将生成一张名为 `video_00_03_00-00_05_00.jpg` 的图像到当前目录。
python ClipToolkit.py [video.mp4] --start-time 00:03:00 --end-time 00:05:00

# 指定 --save-path 则以指定名称保存到指定的目录
python ClipToolkit.py [video.mp4] --start-time 00:03:00 --end-time 00:05:00 --save-path ./output/result.jpg
```

默认情况下，ClipToolkit 导出图像。如希望导出为视频切片，添加 `--mode video` 参数即可：

```sh
# 这将导出一个.avi格式的视频切片
python ClipToolkit.py [video.mp4]  --mode video --start-time 00:03:00 --end-time 00:05:00
```

### 导出检测结果

> **什么是 MDRF？**
> MDRF（Meteor Detection Recording Format）是 MetDetPy 使用的检测结果格式文件。使用 MetDetPy 或 evaluate 运行检测后，会自动生成此文件。MDRF 文件包含视频信息、检测结果的时间戳、检测框坐标和置信度等完整信息。参考[MDRF说明](../data-format-cn.md#流星检测记录格式(MDRF))了解更多。

当需要批量导出检测结果时，将 MDRF 格式文件作为唯一的位置参数即可：

```sh
# 导出所有片段为视频切片到 ./output 目录
python ClipToolkit.py [detection_result.json] --mode video --save-path ./output
```

导出 MDRF 格式文件的检测结果时，可以搭配下列参数改善导出效果：

- **时间补偿**：在片段前后额外添加时间长度，确保目标完整包含在切片中。
- **过滤规则**：过滤掉误检、低置信度或过小的目标，只保留高质量样本。

| 参数 | 说明 |
|------|------|
| `--enable-filter-rules` | 启用过滤规则，参见[过滤规则](#过滤规则)。 |
| `--padding-before` | 片段开始时间前的补偿时间（秒） |
| `--padding-after` | 片段结束时间后的补偿时间（秒） |
|  `--with-bbox` | 在导出的图像/视频中绘制检测框 |

以下示例将同时启用过滤规则、时间补偿和标注框绘制：

```sh
python ClipToolkit.py [detection_result.json] --mode video --with-bbox --enable-filter-rules --padding-before 0.5 --padding-after 0.5 --save-path ./output
```

### 常用参数

| 场景 | 命令 |
|------|------|
| 生成视频切片 | `python ClipToolkit.py [video.mp4] --start-time 00:01:00 --end-time 00:02:00 --mode video` |
| 生成高质量 JPEG 图像 | `python ClipToolkit.py [video.mp4] --start-time 00:01:00 --end-time 00:02:00 --jpg-quality 95` |
| 生成 PNG 格式图像（无损） | `python ClipToolkit.py [video.mp4] --start-time 00:01:00 --end-time 00:02:00 --suffix png` |
| 启用[过滤规则](#过滤规则)（仅导出正样本） | `python ClipToolkit.py [detection_result.json] --enable-filter-rules` |
| 片段前后增加时间补偿 | `python ClipToolkit.py [detection_result.json] --padding-before 0.5 --padding-after 0.5` |
| 启用[去噪处理](#去噪算法) | `python ClipToolkit.py [video.mp4] --start-time 00:01:00 --end-time 00:02:00 --denoise mfnr-mix` |
| 生成带标注框的切片或图像 | `python ClipToolkit.py [detection_result.json] --with-bbox` |
| 生成 MP4 格式视频 | `python ClipToolkit.py [video.mp4] --start-time 00:01:00 --end-time 00:02:00 --mode video --suffix mp4` |

---

## 三种输入模式

ClipToolkit 共支持 3 种输入模式，通过传入的位置参数（Positional Arguments）决定：

### 通用模式

**适用场景：** 需要一次性创建多个图像堆栈或视频切片，并配置每个输出的名称和格式。

**使用方式：** 传入两个位置参数：`target`（目标视频文件路径）和 `json`（JSON 格式的字符串或 JSON 文件路径）

**示例：**
```sh
python ClipToolkit.py ./test/20220413Red.mp4 ./test/clip_test.json --mode image --suffix jpg --jpg-quality 60 --save-path ./test
```

**JSON 格式要求：**
- 必须是一个 JSON 数组
- 每个元素包含：
  - 必需的 `"time"` 键：两个 `"HH:MM:SS.ms"` 格式的字符串数组，表示开始时间和结束时间
  - 可选的 `"filename"` 键：指定文件名和后缀
  - 可选的 `"target"` 键：标注框信息，包含 `"pt1"`、`"pt2"` 和 `"preds"`

**JSON 示例：**
```json
[
  {
    "time": ["00:03:00.000", "00:05:00.000"],
    "filename": "clip1.jpg"
  },
  {
    "time": ["00:10:00.000", "00:12:00.000"],
    "filename": "clip2.jpg"
  }
]
```

### 精简模式

**适用场景：** 仅需处理单个图像堆栈或视频切片，无需构建 JSON 文件。

**使用方式：** 仅传入一个视频文件作为位置参数，起止时间、格式等参数通过可选参数指定。

**示例：**
```sh
python ClipToolkit.py ./test/20220413Red.mp4 --start-time 00:03:00 --end-time 00:05:00 --mode image --save-path ./test/generated_img.jpg
```

**注意：** 如果 `--save-path` 中包含文件扩展名（如 `./test/generated_img.jpg`），程序会自动将其解析为输出文件名。

### 结果批量生成模式

**适用场景：** 为检测结果批量生成所有对应的堆栈图像或视频切片。主要用于搭配 MetDetPy 批量导出检测结果。

**使用方式：** 仅指定一个 `MDRF` 格式的 JSON 文件作为输入（该文件可由 `evaluate` 或 `MetDetPy` 生成）。

**示例：**
```sh
python ClipToolkit.py [video_mdrf.json] --mode video
```

此模式下，程序会根据 MDRF 文件中的检测结果，为每个检测到的目标生成对应的图像或视频切片。

---

## 完整参数说明

```sh
python ClipToolkit.py target [json] [--cfg CFG] [--start-time START_TIME] [--end-time END_TIME]
                              [--mode {image,video}] [--suffix SUFFIX]
                              [--save-path SAVE_PATH]
                              [--jpg-quality JPG_QUALITY] [--png-compressing PNG_COMPRESSING]
                              [--debayer] [--debayer-pattern DEBAYER_PATTERN]
                              [--denoise {mfnr-mix,simple}]
                              [--with-annotation] [--with-bbox]
                              [--enable-filter-rules | --disable-filter-rules]
                              [--padding-before PADDING_BEFORE] [--padding-after PADDING_AFTER]
                              [--debug]
                              [--resource-dir RESOURCE_DIR]
```

### 位置参数

| 参数 | 说明 |
|------|------|
| `target` | 目标视频文件路径或 MDRF 格式 JSON 文件路径 |
| `json` | （可选）JSON 格式的字符串或 JSON 文件路径，包含时间片段信息 |

### 可选参数

#### 基本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cfg`, `-C` | 配置文件路径，管理 ClipToolkit 的默认行为 | `./global/clip_cfg.json` |
| `--mode` | 输出样本模式，从 `{image, video}` 中选择 | `image` |
| `--suffix` | 输出文件后缀（如 `jpg`、`png`、`mp4`、`avi`） | 图像模式 `jpg`，视频模式 `avi` |
| `--save-path` | 输出文件保存路径。单片段时可包含文件名 | 当前工作目录 |
| `--start-time` | 片段起始时间（毫秒数或 `HH:MM:SS.ms` 格式），仅精简模式有效 | 视频起始时间 |
| `--end-time` | 片段结束时间（毫秒数或 `HH:MM:SS.ms` 格式），仅精简模式有效 | 视频结束时间 |

#### 图像质量参数

| 参数 | 说明 | 默认值 | 取值范围 |
|------|------|--------|----------|
| `--jpg-quality` | JPEG 图像质量 | 95 | 0-100 |
| `--png-compressing` | PNG 图像压缩程度 | 3 | 0-9 |
| `--denoise` | [去噪算法](#去噪算法)，从 `{mfnr-mix, simple}` 中选择 | 不启用 | - |

#### 标注相关参数

| 参数 | 说明 | 适用模式 |
|------|------|----------|
| `--with-annotation` | 同时输出 [Labelme 标注](#labelme-标注导出)风格的 JSON 文件 | 通用模式、样本生成模式 |
| `--with-bbox` | 在导出的图像/视频中绘制检测框 | 通用模式、样本生成模式 |

#### 过滤规则参数

| 参数 | 说明 |
|------|------|
| `--enable-filter-rules` | 启用[过滤规则](#过滤规则) |
| `--disable-filter-rules` | 禁用[过滤规则](#过滤规则) |

#### 时间补偿参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--padding-before` | 时间补偿的片段开始时间前的补偿时间（秒） | 配置文件中的 `export.clip_padding.before` |
| `--padding-after` | 时间补偿的片段结束时间后的补偿时间（秒） | 配置文件中的 `export.clip_padding.after` |

#### Debayer参数（仅视频模式）

| 参数 | 说明 |
|------|------|
| `--debayer` | 用于读取raw视频，为每帧应用 Debayer 处理（仅视频模式） |
| `--debayer-pattern` | Debayer 的拜耳阵列模式，如 `RGGB` 或 `BGGR` |

#### 调试参数

| 参数 | 说明 |
|------|------|
| `--debug` | 启用调试模式，打印更详细的日志信息 |

#### 资源目录参数（打包为onefile后运行时配置）

| 参数 | 说明 |
|------|------|
| `--resource-dir`, `-R` | 资源文件夹路径，应包含 `config/`、`weights/`、`resource/` 和 `global/` 子文件夹。指定后从该目录读取静态文件而非默认位置 |

---

## 导出格式支持

| 输出 \ 输入 | RAW | jpg | 延时视频 | 常规视频 |
|-------------|-----|-----|---------|---------|
| 视频 | x | x | x | ✅ 转码，支持 MP4/AVI，带有音频 |
| 视频 + 标注框 | x | x | x | ✅ 转码，支持 MP4/AVI，带有音频 |
| 图像 Only | ✅ 拷贝 | ✅ 拷贝 | ✅ 读+写 jpg | ✅ 读+写 jpg |
| 图像 + 标注框 | ✅ 读+写 jpg | ✅ 读+写 jpg | ✅ 读+写 jpg | ✅ 读+写 jpg |
| 图像 + labelme | ✅ 同步写 | ✅ 同步写 | ✅ 同步写 | ✅ 同步写 |

**图例：**
- x：不支持/不合理的需求
- ✅：支持

**注意：**
- 视频写入器会影响可选的视频后缀。详见[配置文件说明](../config-doc-cn.md#视频写入器配置)。
- 当尝试在 RAW 格式图像上绘制标注框时，会自动保存为 JPG 格式。

---

## 高级功能

### 去噪算法

> **为什么需要去噪？**
> 夜间拍摄的视频通常噪点较多，尤其是高增益设置下。去噪可以显著提升图像质量，使检测目标更清晰。去噪算法还会尝试连接可能的短线，以期生成更美观的图像。

ClipToolkit 支持以下去噪算法：

| 算法 | 参数值 | 说明 |
|------|--------|------|
| 多帧降噪混合 | `mfnr-mix` | 高质量去噪，适用于多帧图像堆叠 |
| 简单降噪 | `simple` | 快速去噪，适用于实时处理 |

**使用示例：**
```sh
python ClipToolkit.py video.mp4 --start-time 00:01:00 --end-time 00:02:00 --denoise mfnr-mix
```


### Labelme 标注导出

通过 `--with-annotation` 参数，可以同时生成 Labelme 风格的标注 JSON 文件。

**使用场景：**
- 为深度学习模型准备训练数据
- 标注数据的人工校验
- 数据集的标准化存储

**使用示例：**
```sh
python ClipToolkit.py video.mp4 detections.json --with-annotation --with-bbox
```

### 过滤规则

> **为什么需要过滤规则？**
> 检测结果文件会包含完整的结果，其中会包含大量误检、低置信度或过小的目标。过滤规则可以自动剔除这些低质量结果，只保留有效的正样本。

当启用过滤规则时（配置文件中设置或命令行指定 `--enable-filter-rules`），ClipToolkit 会根据以下规则过滤目标：

- **类别过滤：** 排除 `DROPPED` 和 `OTHERS` 内置负类，以及配置中指定的排除类别
- **置信度过滤：** 过滤置信度低于阈值的目标
- **尺寸过滤：** 过滤尺寸相对于图像对角线长度比例过小的目标

过滤规则同时应用于图像导出和视频导出。

---

## 配置文件

ClipToolkit 使用配置文件（默认为 `./global/clip_cfg.json`）来设置更多参数。欲了解配置文件结构和字段含义，请查看 [配置文件说明](../config-doc-cn.md)。

---

## 常见问题

### Q1: 如何批量处理多个视频？

**A:** 通用模式支持在一个 JSON 中指定多个片段。如果需要处理多个不同的视频文件，可以使用脚本循环调用 ClipToolkit。

### Q2: 为什么我导出MP4格式的视频会出错？

**A:** 要导出MP4格式的视频，必须要在配置文件中设置 `writer` 为 `FFMpegVideoWriter`。此外，还需要下载 FFMpeg 和 FFProbe 工具，并将配置文件的 `export.ffmpeg_config` 指向放置了上述两个文件的目录。详见[配置文件说明](../config-doc-cn.md#ffmpeg配置)。

### Q3: 为什么我的视频切片没有音频？

**A:** 确保使用支持音频的 VideoWriter（如 FFMpegVideoWriter），并在配置文件中正确配置。OpenCVVideoWriter 不支持音频。

### Q4: 过滤规则如何影响标注导出？

**A:** [过滤规则](#过滤规则)会先过滤掉不符合条件的目标，然后仅对保留的目标进行标注框绘制和 [Labelme 标注](#labelme-标注导出)导出。

### Q5: 如何启用 Debayer 处理？

**A:** 使用 `--debayer` 参数启用 [Debayer 处理](#debayer参数)，并使用 `--debayer-pattern` 指定 Bayer 模式（如 `RGGB`、`BGGR`）

### Q6: 为什么文件名中的时间冒号被替换成下划线？

**A:** 为了兼容不同操作系统，ClipToolkit 会自动将文件名中的冒号 `:` 替换为下划线 `_`。例如：`00:03:00-00:05:00` 会变成 `00_03_00-00_05_00`。

---

## 相关文档

- [工具使用说明（中文）](../tool-usage-cn.md)
- [配置文件说明](../config-doc-cn.md)
- [流星检测记录格式 (MDRF)](#meteor-detection-recording-format-mdrf)
