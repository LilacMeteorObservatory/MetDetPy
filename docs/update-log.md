
# Update Log

## Todo List

1. 实时模式
    1. 支持直接使用流数据源输入，如RTMP/RTSP
    2. 支持断连和自动重连功能
    3. 支持流式的fps估算（代替目前的采样估算）
2. 检测效果（正负样本筛选）
    1. 引入对轨迹的序列描述特征（如通过抽样得到序列），排除飞虫类/云类非规则运动。
    2. 改善检测卫星等飞行器是容易断线的不连续性问题（可能需要对检测底层机制进行修改）。
    3. 对于传统过滤器，引入简单分类器（如SVM或ANN）代替目前的特征阈值卡控机制。
    4. 继续微调检测模型以改善准确率(增训，微调)，细化精灵类标注，增加对其他天象（如闪电）的支持。
    5. 感知未知类型（即非干扰但也非目标的检测结果,常见于各类再入，不在分类器中的大气现象等）的现象并记录。
    6. 使模型检测器输出带上流星方向信息，为后续迭代建设基础。
    7. 对噪声估计不准确的场景引入自动化微调参数，改善该类型场景对暗弱流星的检测。
    8. 优化/修复大面积类响应被忽略的问题。
    9. 添加"焦距"配置，为鱼眼/长焦的流星视频的判据提供修正值。
    10. 添加指示亮度的参数项。
    11. 接入最小二乘损失项作为特征。
    12. 接入动态尺度或者多尺寸检测。
    13. 低反差目标？
    14. 对特大目标的检测效果确认。
3. 检测性能
    1. 简化迭代中产生的冗余检测后处理机制（如DyMask， MeteorLib中的冗余逻辑），提升检测性能。
    2. 构建直接使用差值代替线段检测的高速检测器，降低计算量，并预期能改善城市场景卫星/飞机的问题。
    3. 将计算复杂的模块使用cython等实现，改善性能
4. 功能扩展
    1. 接入对其他深度学习框架和模型的支持
    2. 添加天区解析功能，为支持快速叠图，分析辐射点，流星组网监测提供基础支持
    3. 支持导出UFO Analizer格式的文本，用于流星组网联测等需求
5. bugfix
    1. 单帧长度指定大于窗口允许时，差值不生效问题（部分相机可能取到低值）。
    3. 部分视频片段的fps问题
    
6. 其他杂项优化
    1. 日志中的error需要写到stderr中去。
    2. 其他代码中的TODO项

Note:
1. 炸屏优化（1. 天亮/天暗时的亮度突变过滤；2.火流星等直接炸屏的场景检出）
2. 飞行器/大面积时间不容易被后验检测器检出。
3. 改善报错信息（尤其是配置文件）


## Version 2.3.1

✅ Improvement
* 为 `VideoLoader` 添加 `continue_on_err` 选项，设置为 `true` 时会在帧数据解析失败时继续尝试向后解析[1]。
*  `ClipToolkit` 生成截图失败时，会在 `stderr` 抛出 fatal error，便于定位问题。
* 冗余代码优化。

⚠️1：该设置可能不能改善缺失关键帧索引造成的连续读取失败。如果您的视频频繁出现大量失败，请优先尝试重新封装视频：
```ffmpeg -i input.mp4 -c copy -map 0 fixed.mp4```。

✅ BugFix
* 修复了数个配置文件键值错误问题。
* 修复部分情况下 `np.int32` 类型数据导致的序列化失败问题。
* 延长了 `metlog` 组件允许的最长等待时间，优化可能的输出不完整问题。

TODO: pyexiv2 pkging.

## Version 2.3.0

✅ New Feature(s)
1. 新增视频解码后端 `PyAVVideoWrapper` ，并替代`OpenCVVideoWrapper`作为默认视频解码后端。相比`OpenCVVideoWrapper`，`PyAVVideoWrapper`在跳转准确性和稳定性上表现更好，改善读取时间索引错位视频的准确性，同时修复了视频切片和重校验的时间错位问题。

2. 为 `ClipToolkit` 新增 `--denoise` 选项，支持在导出图像时降低图像背景噪声水平，同时连接图像中前景目标可能存在的断线。可以指定 `simple` （对最大值图像简单降噪）或者 `mfnr-mix` （使用“最大值-平均值混合”叠加算法的多帧降噪）两种降噪算法*。


⚠️ *：`simple` 是使用双边滤波的传统降噪算法，运行较快；`mfnr-mix`是混合多帧降噪结果的视频降噪算法，效果更好，但耗时更长（通常慢于直接叠加5倍以上），适用于需要导出较高画质的场合。

✅ Improvement(s)
1. `ClipToolkit` 在保存图像时，会嵌入 `sRGB` 色彩配置文件。

✅ Bug Fixed
1. 修复命令行直接调用`ClipToolkit`的`--save-path`时，会忽略格式直接新建文件夹的问题。
2. 修复`ClipToolkit`绘制包围框时的效果问题。
3. 修复`MetDetPy`在重校验失败时直接跳过输出片段的问题。
4. 修复`MetDetPy`在重校验时可能会无法生成截图的问题。
5. 修复`MetDetPy`在调用深度学习模型时可能出现非预期错误导致检测终止的问题。
6. 修复 `MetDetPhoto` 检测单张照片时可视化的问题。

✅ Code Formatting
1. 引入 `typing` 规范类型注解和静态类型检查。
2. 引入 `dacite` 管理配置文件规范和参数解析。

## Version 2.2.0

✅ New Feature(s)

1. 更新默认检测模型为 `yolov5s_v2.onnx`，改善卫星线等场景的准确率，并支持了飞虫、喷流等更多检测类别。
2. 重构并完善用于图像检测的工具入口 `MetDetPhoto`。该工具现可用于单张照片，延时序列或者延时序列构成的视频检测。
3. `Model` 相关配置项支持多尺度推理相关设置，可以改善小目标召回率。
4. 正式定义流星检测结果格式(MDRF)作为MetDetPy的流转文件格式，增加各脚本的通用性。该格式文件可以手动创建（作为标注），从检测/评估工具导出，或作为评估和切片工具的输入。有关该脚本的更多介绍见[工具文档](./tool-usage-cn.md)/[Tool Usage](./tool-usage.md)。`MetDetPy` 和 `MetDetPhoto` 可以输出MDRF文件，而`ClipToolkit` 和 `evaluate` 则可以接受MDRF文件的输入。

✅ Improvement(s)

1. 调整了重校验阈值与最终预测的得分的计算方式，改善了非直线目标（如红色精灵等大气现象）的召回性能。
2. 改进了 `ClipToolkit` 的易用性，主要优化点有：（1）提供了简化的调用接口，支持了无参数（完整视频）及单张截图时的参数；（2）允许同时保存Labelme格式标注；（3）支持保存带标注框的图像；（4）导出视频时以流式保存，避免内存溢出。
3. 调整了后端模式下的输出逻辑，未被分类为正样本的检测结果也会在后端输出（同样以 Meteor 开头）。
4. 增加了归一化的速度(fix_speed)，距离(fix_dist)，持续时间(fix_duration)和持续运动时间(fix_motion_duration)的统计值输出，为统计值在不同设备上的统计范围提供了统一参考。
5. 优化了 `evaluate` 工具接受的参数列表，完善了回归测试与性能测试的数据格式和基线。
6. 调整保存到检测结果文件的坐标与格式更改为与命令行输出一致（即视频原始分辨率下的坐标）,提高输出结果易用性。

✅ Bug Fixed

1. 使用归一化的速度统计值进行流星置信区间的判定，预期改善对于卫星和飞机的误判。

⚠️ 本次更新对默认参数范围有较大变动，使用从先前版本复制的自定义配置文件可能在新版本不适用，建议用户参考参数解释页手动更新配置文件。


## Version 2.1.0

✅ New Feature(s)
1. 为发行版引入 `onnxruntime_directml`，修复 onnxruntime 在 windows 平台不支持使用GPU的问题，缓解CPU负载。
2. 新增 `--live-mode` 选项，开启该选项可使检测速度与视频时长能够基本持平，均衡CPU负载，适用于直播场景。
3. 新增 `--save` 选项，允许保存运行结果到json文件。

✅ Improvement(s)
1. 更新了评估工具`evaluate.py`，更改了评估标注格式，支持保存结果，与其他版本比较性能和效果等。该结果可作为回归测试与性能测试的基准，以确保迭代间的性能和效果稳定性。

⚠️ 提供了 `--provider cpu` 以指定CPU运行模型，如果在未安装 DirectX 12的电脑上使用 `onnxruntime_directml` 造成报错，可以作为降级方案。

✅ Bug Fixed
1. 为Stacker增加出错兜底逻辑，修复Stacker在特定场景下报错会导致序列后续无输出的问题。
2. 修复了2.0.1版本因为动态模板机制出错产生的效果差异。
3. 通过放宽重校验机制，临时修复了对可能火流星的误报。后续会对该类问题给出更稳健的解决方案。

⚠️ Revert
1. 回滚了2.0.2版本中有关动态间隔机制的改动，维持效果一致性。该机制的下线还需要进一步测试。


## Version 2.0.2

✅ Modification
* 修改了响应合并逻辑及输出概率的计算逻辑，改进了输出的预测概率。
* 更改了后处理阶段的线段合并机制。
* 下线了动态间隔机制。

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
