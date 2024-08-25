
# Update Log


### TODO for v2.1.1

1. 实时模式
    1. 支持直接使用流数据源输入，如RTMP/RTSP
2. 检测效果（正负样本筛选）
    1. 引入对轨迹的序列描述特征（如通过抽样得到序列），排除飞虫类/云类非规则运动。
    2. 改善检测卫星等飞行器是容易断线的不连续性问题（可能需要对检测底层机制进行修改）。
    3. 对于传统过滤器，引入简单分类器（如SVM或ANN）代替目前的特征阈值卡控机制。
    4. 继续微调检测模型以改善准确率(增训，微调)，及增加闪电支持。
    5. 感知未知类型（即非干扰但也非目标的检测结果,常见于各类再入，不在分类器中的大气现象等）的现象并记录。
    6. 引入回归测试，确保迭代间的效果稳定性，量化效果迭代收益。
    7. 使模型检测器输出带上流星方向信息，为后续迭代建设基础。
3. 检测性能
    1. 简化迭代中产生的冗余检测后处理机制（如DyMask， MeteorLib中的冗余逻辑），提升检测性能。
    2. 构建直接使用差值代替线段检测的高速检测器，降低计算量，并预期能改善城市场景卫星/飞机的问题。
    3. 对噪声估计不准确的场景引入微调参数，改善该类型场景对暗弱流星的检测。
    4. 引入性能测试，确保迭代间的性能稳定性，量化性能迭代收益。
4. 已探明的问题修复
    1. 期望修复时间索引错位问题。
5. 其他杂项优化
    1. 引入叠加支持最大值-平均值混合叠加，改善叠加画质。
    2. 日志中的error需要写到stderr中去。
    3. 其他TODO


## Version 2.1.0

✅ Bug Fixed
1. 为Stacker增加出错兜底逻辑，修复Stacker在特定场景下报错会导致序列后续无输出的问题。
2. 修复了2.0.1版本因为动态模板机制出错产生的效果差异。

✅ Improvement(s)
1. 为发行版引入 `onnxruntime_directml`，修复 onnxruntime 在 windows 平台暂时不支持使用GPU的问题，缓解CPU负载。
2. 新增 `--live-mode` 选项，开启该选项可使检测速度与视频时长能够基本持平，适用于直播场景。
3. 更新了评估工具`evaluate.py`，更改了评估标注格式，支持保存结果，与其他版本比较性能和效果等。

⚠️ 提供了 `--provider cpu` 以指定CPU运行模型，如果在未安装 DirectX 12的电脑上使用 `onnxruntime_directml` 造成报错，可以作为降级方案。

## Version 2.0.2

✅ Modification
* 修改了响应合并逻辑及输出概率的计算逻辑，改进了输出的预测概率。
* 更改了后处理阶段的线段合并机制。

✅ Improvement(s)
* 为loader增加了“曝光时间上限”（upper_bound）选项（非必须项）。
* 补充了直接运行模型(run_model.py)的支持。

## Version 2.0.1

✅ Bug Fixed
* 输出编码统一为utf-8。
* 修复make_package中的错误。
* 完善了视频读取非正常退出时的结束逻辑。

✅ Modification
* 可视化接口重构及通用性扩展。

✅ Improvement(s)
* 优化了deetector与collector中的参数计算逻辑，加速检测。

## Version 2.0.0

✅ Bug Fixed
* 修复输出时间错位的问题
* 修复ClipToolkit对单个输入的支持

✅ New Feature(s)

* 支持深度学习检测器：可使用ONNX框架下YOLO格式的网络作为检测器
* 支持“重校验”选项：预测结果可以通过神经网络进行重校验，降低误报率
* 预测结果新增类别，可支持对流星以外的天文或大气现象进行检测

✅ Modification

* 大量代码和API重构
* config格式调整；不再兼容v1.x版本的config文件
* 引入多种可选预设替代简单的"sensitivity"设置
* 可视化接口分离
* 项目许可证更换为MPL2.0许可证

✅ Improvement(s)
* 优化部分预设参数，改善默认检测模式下的运行效果。
* 优化多线程视频读取和日志模块设计，完善退出时线程回收逻辑。
* 优化估算SNR的计算流程。
* 优化make_package在多平台和复杂环境下的打包稳定性。

## Version 1.3.0

✅ Bug Fixed

* 修复了直线检测获取直线的数目问题，修复了多个流星同屏时的漏报问题。
* 为流星收集器中引入“面积”类，改善了直线检测算法下对火流星的漏报的情况。
* 其他的已知问题。

✅ API改进

* 命令行启动时的start-time与end-time参数现支持输入`HH:MM:SS.MS`格式的时间。
* 调整检测的返回值：返回的JSON中的`pt1`与`pt2`将按顺序描述起止位置。
* Some API of [config.json](../config.json) is modified and going to be merged or deprecated in the future. To keep compatibility, old API is reserved until `v2.0.0`. / 对 [config.json](../config.json) 的部分API进行了修改，先前版本的部分接口会在未来被合并或废弃。为保持兼容性，旧版本API会被保留至`v2.0.0`。
* Some API of [config.json](../config.json) is modified and going to be merged or deprecated in the future. To keep compatibility, old API is reserved until `v2.0.0`. / 对 [config.json](../config.json) 的部分API进行了修改，先前版本的部分接口会在未来被合并或废弃。为保持兼容性，旧版本API会被保留至`v2.0.0`。

✅ 算法改进

* 简化了二值化前的图像增强方式
* 增加了对检出直线的质量评定与非最大值抑制（NMS）机制
* 增加了自适应gap
* 增加了动态遮罩机制，优化了相机直录视频在星点附近的噪声
* 引入面积响应，改善了容易排异火流星的情况
* 完善了灵敏度相关的预设

⚠️ 这些改进主要针对“标准”灵敏度，能够优化大量的误报，但会导致对一些暗弱流星的召回率降低。如果希望获得更高的召回率，可以在config中关闭对应设置，或启用灵敏度为“高”。

✅ 评估脚本正式上线

* 评估脚本能够自动根据标准加载视频，掩模以及标注，为对算法性能的定量分析提供支持。

## Version 1.2.4

✅ Fix bugs about MultithreadMetLog. / 修复了一些 `MultithreadMetLog` 模块的问题。

✅ Some API of [config.json](../config.json) is modified and going to be merged or deprecated in the future. To keep compatibility, old API is reserved until `v2.0.0`. / 对 [config.json](../config.json) 的部分API进行了修改，先前版本的部分接口会在未来被合并或废弃。为保持兼容性，旧版本API会被保留至`v2.0.0`。

## Version 1.2.3

✅ [ClipToolkit.py](../ClipToolkit.py) related API is redesigned and updated. / 对 [ClipToolkit.py](../ClipToolkit.py) 脚本的API进行了重新设计与更新，简化了使用，提供了更多选项。

✅ Introduce `MultithreadMetLog` module to manage output and logging. / 引入了 `MultithreadMetLog` 模块管理输出与日志。

## Version 1.2.2

✅ [ClipToolkit.py](../ClipToolkit.py) is available now. This script can be used to create several video clips or stacked images at once. See its usage for details. / 增加了 [ClipToolkit.py](../ClipToolkit.py)工具。该工具可用于一次性创建多个视频剪辑切片或堆栈图片。可参考其用法获取更多信息。

✅ Update packaging script: modify the compile option to accelerate compiling speed(nuitka only); support using UPX to compress the size of executables;  support package multiple executables once. /更新打包脚本：修改编译选项以加速编译（Nuitka）；支持UPX压缩选项减小可执行文件大小；支持同时打包多个程序。

✅ Bug fixed and code optimization. / 修复Bug，优化代码结构

⚠️ Since this version, the output location of meteors is changed from running resolution to raw resolution. To keep compatibility, `video_size` is provided in the output JSON. / 从该版本开始，输出的流星位置将会从运行时分辨率调整为原始分辨率。为保持兼容性，输出原始分辨率的JSON会额外提供`video_size`键值。

## Version 1.2.1

✅ Bug fixed:
    Fix the exception output of the start time under some conditions / 修复了部分情况下起始时间异常的问题；
    Fix the exception abort issue during the FPS estimation / 修复了帧率估算时异常结束的问题；
    Fix the infinite loop issue during the FPS estimation / 修复了帧率估算时Sigma裁剪均值无法收敛的问题.

✅ Update packaging script: add Nuitka option, update spec file of pyinstaller. / 更新打包脚本：增加Nuitka打包选项，更新pyinstaller使用的.spec文件。

✅ Formally release packaging version for linux and macOS platforms. / 正式发布macOS和linux平台的打包版本。

## Version 1.2

✅ Resolution-related APIs have been improved to support videos with different aspect ratios / 优化了分辨率相关接口以支持不同长宽比的视频

✅ Implemented an adaptive binary threshold mechanism. We explored simple empirical relations between std and the binary threshold. (⚠️This may not be appropriate for all video patterns. If you encounter any issues, please feel free to modify the function or help me improve it:) / 实现根据方差计算的自适应二值化阈值

✅ Sensitivity configuration is supported now: you can modify "sensitivity" under "bi_cfg" in config.json or passing arguments "--sensitivity {option}" to the program. / 增加灵敏度设置

✅ Smoothing probabilities to [0,1] instead of {0,1}. Meteor judging now is by a series of factors so it could be more accurate (perhaps...). / 输出调整为概率形式

## Version 1.1.1

✅ Improved non-ASCII filename support for packaging version /改善对非ASCII文字路径的支持: by packaging with Python 3.7 and later, the new `pyinstaller` (>=5) can support non-ASCII paths and filenames.

## Version 1.1

✅ Add "Backend" mode to support MeteorMaster. / 增加了后端模式

✅ Update output stream format. / 调整输出流格式

## Version 1.0

Initial Version / 初始版本
