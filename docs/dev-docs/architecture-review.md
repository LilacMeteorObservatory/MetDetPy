# MetDetPy 架构审查报告

> 审查时间：2026-05-29  
> 基于版本：V2.4.0 (dev branch, commit 33a6b35)

## 二、关键缺陷清单

### TODO-ARCH-003: Exporter 线程安全（P1）

**状态**: 未修复  
**位置**: `MetLib/collector.py` — `MetExporter`

| 子问题 | 描述 |
|--------|------|
| 003-a | `self.meteor_list` 在后台线程 append，主线程无锁读取 |
| 003-b | Exporter loop 未 catch 未知 flag 的 KeyError，daemon 线程死亡无通知 |
| 003-c | `ThreadMetLog` 的 `stopped` + `is_empty` 检查无同步原语 |

**建议**: 为 `meteor_list` 添加 threading.Lock；exporter loop 添加顶层 try/except 并记录异常。

---

### TODO-ARCH-004: dacite Union 反序列化不可靠（P1）

**状态**: 未修复  
**位置**: `MetLib/metstruct.py`

**描述**: `DetectorCfg.cfg: Union[BinaryCfg, DLCfg, BrightnessCfg]` — dacite 按顺序尝试，字段子集重叠时可能选错类型。同理 `MDRF.basic_info` 和 `MDRF.results` 的 Union。

**建议**: 引入 discriminated union 模式 — 在 `from_json_file` 中根据 `detector.name` 手动选择目标类型再调用 dacite。

---

### TODO-ARCH-005: VideoLoader God Constructor（P1）

**状态**: 未修复  
**位置**: `MetLib/videoloader.py` — `VanillaVideoLoader.__init__`

**描述**: 构造器同时执行 mask加载、管线构建、视频打开、曝光估算（实际读帧并多次reset/start/stop）、参数校验。曝光估算中修改自身状态 — 在构造完成前对象经历多次状态转换。

**建议**: 将曝光估算提取为独立工厂方法或 builder 阶段，构造/估算/启动三阶段分离。

---

### TODO-ARCH-006: 全局可变单例（P1）

**状态**: 未修复  
**位置**: `MetLib/utils.py`, `MetLib/metlog.py`

| 子问题 | 描述 |
|--------|------|
| 006-a | `ID2NAME`, `NAME2ID`, `NUM_CLASS` 模块级可变全局，懒加载时机依赖调用顺序 |
| 006-b | `met_logger` 单例 + `set_default_logger` 修改模块级 `level_header` 内容（backend模式改名） |

**建议**: class_name 改为 config 阶段加载一次后通过 RuntimeParams 或 context 对象传递；logger 改为实例持有。

---

### TODO-ARCH-007: DictAble.to_dict 继承缺陷（P1）

**状态**: 未修复  
**位置**: `MetLib/metstruct.py` — `DictAble._key2value` / `to_dict`

**描述**: 使用 `self.__annotations__.keys()` 只遍历当前类直接定义的字段，继承链上的字段被遗漏。

**建议**: 改用 `dataclasses.fields(self)` 遍历所有字段。

---

### TODO-ARCH-008: 错误处理不一致（P1）

**状态**: 未修复  
**位置**: 全局

| 模式 | 位置 | 行为 |
|------|------|------|
| 硬杀进程 | collector.py `prob_meteor` | `exit()` |
| 静默吞错 | PyAVVideoWrapper.read() | 捕获所有异常返回 `(False, None)` |
| 无异常处理 | Exporter loop | daemon 死亡无通知 |
| 合理降级 | `recheck_progress` | stacking 失败时跳过 |

**建议**: 定义 `MetDetError` 异常层级；边界处 catch + 日志 + 降级传递，内部 let-it-crash + 上层统一兜底。

---

### TODO-ARCH-009: Sentinel 对象反模式（P2）

**状态**: 未修复  
**位置**: `MetLib/collector.py` — `MeteorCollector.__init__`

**描述**: `self.met_active` 初始化为填充 magic value（`2**16`, `np.nan`, `None`）的虚假 `MeteorSeries`，所有迭代逻辑必须容忍首元素为垃圾数据。

**建议**: 改为 `Optional[MeteorSeries]` 或使用空列表。

---

### TODO-ARCH-010: MDRF 嵌入完整 config（P2）

**状态**: 未修复  
**位置**: `MetLib/metstruct.py` — `MDRF.config`

**描述**: 输出格式直接嵌入 `MainDetectCfg`，配置 schema 变动即破坏已有 MDRF 文件向后兼容。无迁移逻辑。

**建议**: config 部分仅保存关键参数摘要，或引入 MDRF schema version + 向后兼容解析。

---

### TODO-ARCH-011: MDTarget.exclude_attrs 混合关注点（P2）

**状态**: 未修复  
**位置**: `MetLib/metstruct.py` — `MDTarget`

**描述**: 序列化策略（哪些字段不导出）作为 `default_factory` 嵌入每个数据实例，违反关注点分离。

**建议**: 改为 `ClassVar` 或序列化器外部参数。

---

## 三、算法设计审查

### 3.1 Detector 阶段

#### M3Detector 帧差法

**机制**: 在滑动窗口内计算 `diff = max(frames) - mean(frames)`，3x3 中值滤波后二值化，再做 Hough 直线检测。

**缺陷**:

| ID | 问题 | 影响 |
|----|------|------|
| TODO-ALGO-001 | 慢速目标（持续覆盖大部分窗口帧）同时抬高 max 和 mean，差值趋零，漏检 | 对卫星/飞机漏报 |
| TODO-ALGO-002 | `refresh_max()` 每帧重算 `np.max(sliding_window, axis=0)` — O(n×H×W)，性能瓶颈 | 高分辨率高窗口时延迟大 |
| TODO-ALGO-003 | 帧差 500 条线时整帧丢弃，云层运动或抖动场景导致频繁时间间隙 | 间隙期流星漏检 |

#### 自适应阈值

**机制**: `SNR_SW` 周期性计算子区域标准差，EMA 平滑（60s有效窗口），经二次函数 `1.2x² + 3.6`（normal）映射为二值化阈值。

**缺陷**:

| ID | 问题 | 影响 |
|----|------|------|
| TODO-ALGO-004 | 命名误导 — 计算的是噪声水平（std），非信噪比（SNR） | 代码可读性差 |
| TODO-ALGO-005 | 子区域选择贪心向上移动10px以最大化mask覆盖，可能选到天空角落非典型噪声区 | 阈值估计偏差 |
| TODO-ALGO-006 | 二次映射下 std>2 时阈值 >8，抑制暗弱流星 | 暗弱目标召回率降低 |
| TODO-ALGO-007 | 周期更新（每 `nz_interval*n` 帧），突变噪声时阈值滞后 | 云过境等场景短暂误报增加 |

#### 动态掩模（Dynamic Mask）

**机制**: 额外滑窗跟踪二值化激活图，像素在所有帧中都被激活则标记为掩模，配合腐蚀运算。

**缺陷**:

| ID | 问题 | 影响 |
|----|------|------|
| TODO-ALGO-008 | 间歇性响应（仅缺1帧就逃逸抑制）无法被过滤 | 热噪点闪烁时失效 |

#### BrightnessDetector

**机制**: 网格分区，EMA追踪基线+方差，z-score > 阈值 AND 绝对变化量 > 阈值时触发，邻接合并。

**缺陷**:

| ID | 问题 | 影响 |
|----|------|------|
| TODO-ALGO-009 | EMA 在检测后更新，持续亮度事件逐步抬高基线自行抑制 | 慢速变亮（如天亮）可能持续触发至收敛 |

#### MLDetector

**机制**: 滑窗 max 投影 → 整帧送入 YOLO，逐帧检测。

**缺陷**:

| ID | 问题 | 影响 |
|----|------|------|
| TODO-ALGO-010 | 每个检测周期整帧送模型，无 ROI 候选机制减少计算 | 高FPS时计算量线性增长 |

---

### 3.2 Collector 阶段

#### 运动追踪/关联

**机制**: 新检测线段对现有活跃序列做最近邻关联 — 任意端点在 `max_acceptable_dist` 内 + 时间在 `max_acti_frame` 内即匹配。

**缺陷**:

| ID | 问题 | 影响 |
|----|------|------|
| TODO-ALGO-011 | 贪心首匹配，无全局最优分配 | 两条空间接近的轨迹可能串扰 |
| TODO-ALGO-012 | Track merge 仅基于时间近邻，无空间连续性检查 | 时间接近但空间无关的响应可能被错误合并 |

#### 概率/置信度计算

**机制**: 梯形函数乘积模型：`P = P_time × P_speed × P_len × P_drct`，每个因子在"理想范围"外线性衰减至零。

**缺陷**:

| ID | 问题 | 影响 |
|----|------|------|
| TODO-ALGO-013 | 纯乘法模型 — 单一因子轻微偏离即显著压低整体得分 | 噪声帧导致方向抖动→整体得分大幅下降 |
| TODO-ALGO-014 | 所有阈值参数固定（非自适应），仅速度/距离做了分辨率归一化 | 不同FPS/噪声水平下同一组参数表现差异大 |

#### Recheck 阶段

**机制**: max-stacking 生成叠加图 → YOLO 推理 → IoU 匹配（阈值0.5）→ 分数融合。

**缺陷**:

| ID | 问题 | 影响 |
|----|------|------|
| TODO-ALGO-015 | IoU 0.5 硬阈值无回退；暗弱短流星叠加足迹与运行时bbox差异大时匹配失败 | 弱目标被静默丢弃 |
| TODO-ALGO-016 | 面积型响应方向计算使用 bbox 对角线方向，几何上无意义 | 对方向一致性 feature 引入纯噪声（实际影响有限，因走不同评分路径） |

---

### 3.3 Model 推理阶段

#### 多尺度预测

**机制**: 将图像分割为 `h_rep × w_rep` 个重叠 tile，独立推理后合并结果，再做跨 tile NMS。

**缺陷**:

| ID | 问题 | 影响 |
|----|------|------|
| TODO-ALGO-017 | stride/size 公式不保证完全覆盖图像边缘，右下角像素可能遗漏 | 边缘目标漏检（低概率） |
| TODO-ALGO-019 | 分数计算 `sqrt(conf * cls_prob)` 非标准几何均值，小值放大效应 | 阈值边界附近行为难以预测 |
| TODO-ALGO-028 | `yolov5s_v2.onnx` 元数据含 `id=8: others`，但 `class_name.txt` 仅定义 0–7，运行时会把 id 8 映射为 `DROPPED` | `OTHERS` 样本少且通常不是期望目标，暂不处理，降为低优先级 |

#### 线程安全

| ID | 问题 | 影响 |
|----|------|------|
| TODO-ALGO-020 | Lock timeout (5s) 返回 False 后未检查，直接继续推理 | ONNX session 并发访问崩溃风险 |

---

### 3.4 图像处理/特征阶段

#### imgproc Transform

| ID | 问题 | 影响 |
|----|------|------|
| TODO-ALGO-021 | `scale2tgt_mean`: 全黑帧 `l_gray_mean=0` → 除零产生 Inf/NaN | 后续管线崩溃 |
| TODO-ALGO-022 | 无逆变换机制，坐标映射需外部跟踪 | 增加上下游耦合 |

#### feature.py 特征提取

| ID | 问题 | 影响 |
|----|------|------|
| TODO-ALGO-023 | `calc_roi_gradient` 输入 [0,π] 但输出 `% 2π` → 域不一致 | 梯度方向特征可能返回超出预期范围的值 |
| TODO-ALGO-024 | `calc_brightness_with_roi`: Otsu 全黑/全白时 `np.mean(empty_array)` 返回 NaN | NaN 传播到后续评分 |

#### stacker.py 叠加算法

| ID | 问题 | 影响 |
|----|------|------|
| TODO-ALGO-025 | `mfnr_mix_stacker` 中 `get_gumbel_mean(n=1)` → `sqrt(2*log(1))=0` → 除零 | 单帧输入时崩溃 |
| TODO-ALGO-026 | `np.average(arr[arr>0])` 在无正值时返回 NaN → 前景mask计算失败 | 极短序列或极暗序列时异常 |
| TODO-ALGO-027 | `highlight_preserve >= 1.0` 配置时 `1/(1-x)` 除零 | 配置校验缺失 |

---

## 五、功能 Todo 优先级重排

基于架构健康度评估，对 `update-log.md` 中功能类 Todo 重新排序：

### P2 — 效果/性能类

| ID | 项目 | 备注 |
|----|------|------|
| TODO-FEAT-001 | 低 fps 表现差 | `window_sec * eq_fps < 3` 让 M3Detector 不可用，需算法层支持 |
| TODO-FEAT-002 | 相机直录表现差 | 曝光估算经验公式对直录不准，依赖 ARCH-005 重构 |
| TODO-FEAT-003 | 飞机分类算法 | 需 Collector 支持可插拔 filter chain |
| TODO-FEAT-004 | 亮度突变优化 | BrightnessDetector 需完善阈值和事件类型区分 |
| TODO-FEAT-005 | 并行图像检测 | model 推理并行化需解决 ONNX session 线程安全 |

### P3 — 中长期规划

| ID | 项目 | 备注 |
|----|------|------|
| TODO-FEAT-006 | 天区解析 | 大功能，需外部库，建议独立模块 |
| TODO-FEAT-007 | RTMP/RTSP流支持 | PyAV已支持，主要是断连重连+流式fps估算 |
| TODO-FEAT-008 | TensorRT/CUDA支持 | onnxruntime TensorRT EP + 模型格式转换 |
| TODO-FEAT-009 | 焦距配置 | 在 collector 速度/长度判据中引入缩放因子 |
| TODO-FEAT-010 | 导出得分设置 | 已有 aesthetic_score，需暴露阈值配置 |
| TODO-FEAT-011 | 自动化蒙版 | - |
| TODO-FEAT-012 | 深度学习模型更新（数据层） | - |

---

## 六、推荐重构路径

按优先级排序的执行顺序：

1. **修正确性 Bug**（ARCH-001, ARCH-002, ARCH-007, ALGO-020, ALGO-021, ALGO-025）— 直接导致崩溃或数据错误
2. **数值健壮性**（ALGO-024, ALGO-026, ALGO-027）— NaN/除零传播链修复，添加输入校验
3. **引入 discriminated config dispatch**（ARCH-004）— 消除配置解析隐患
4. **VideoLoader 阶段化重构**（ARCH-005）— 为 FEAT-001/002 提供修复基础
5. **统一错误处理模型**（ARCH-008）— 引入异常层级，主入口 try/finally 兜底
6. **消除全局可变状态**（ARCH-006）— class_name/logger 改为依赖注入
7. **算法改进**（ALGO-013 置信度模型鲁棒化, ALGO-018 NMS阈值调优, ALGO-002 max增量维护）— 效果与性能提升

---

## 七、实验性检测器评估：DiffAreaGuidingDetector

> 位置：`MetLib/Detector.py` L451-519（类名 `DiffAreaGuidingDetecor`，原文拼写）

### 7.1 设计意图与当前状态

该检测器是从 M3Detector 实践缺陷中提出的实验性方案，设计管线为：

```
EMA背景维护 → 直方图分位阈值(卡尔曼滤波) → 二值化+噪点滤波+动态掩模 → 连通域区域检测 → 输出
```

**当前实现状态：原型/不可用。** `detect()` 始终返回 `[], []`，仅第 1 步（EMA 背景）被实现，且阈值逻辑为占位代码：

```python
# 实际含义: frame > ema_bg - 100，对正常天空像素几乎恒真
self.diff_img = ((self.cur_frame.astype(np.float64) + 100) > self.bg_maintainer.cur_value)
```

设计描述的直方图阈值、卡尔曼滤波、连通域检测均未实现。

### 7.2 计算效率对比

| 维度 | M3Detector | DiffAreaGuiding (设计意图) | 效率增益 |
|------|-----------|--------------------------|----------|
| 背景维护 | SlidingWindow `refresh_max`: O(n×H×W) 每帧 | EMA: O(H×W) 每帧 | **n 倍**（n=窗口帧数，通常3-8） |
| 内存占用 | 2×n×H×W (主滑窗+子滑窗) + dy_mask滑窗 | 1×H×W (EMA float64) | **数倍减少** |
| 阈值计算 | 周期性子区域 std + 二次函数 | 逐帧直方图分位 + 卡尔曼 | 近似（均为 O(H×W)） |
| 前景提取 | HoughLinesP (复杂度不稳定，受边缘密度影响) | connectedComponents O(H×W) | **更稳定** |

**结论**：理论计算效率显著优于 M3Detector，特别是消除了 ALGO-002 的 O(n) 性能瓶颈。

### 7.3 对 M3Detector 缺陷的解决能力

| M3Detector 缺陷 | 能否解决 | 分析 |
|-----------------|---------|------|
| ALGO-001: 慢速目标 max≈mean 差值趋零 | **部分** | EMA 背景对慢速目标的记忆有指数衰减，`1/(1-momentum)` 帧后目标从背景中浮现；但极慢目标（持续远超衰减周期）仍被吸收 |
| ALGO-002: refresh_max O(n×H×W) 瓶颈 | **完全解决** | EMA 更新为 O(H×W) |
| ALGO-003: 500线强制丢帧 | **完全解决** | 不使用 Hough 直线，改用连通域，无此问题 |
| ALGO-005: 噪声子区域选择偏差 | **可能解决** | 直方图分位法天然全图自适应，不依赖子区域采样 |
| ALGO-006: 暗弱目标阈值过高 | **取决于实现** | 固定面积比例理论上能恒定灵敏度，但有新矛盾（见下文） |
| ALGO-007: 阈值更新滞后 | **改善** | 卡尔曼滤波逐帧更新，无周期延迟 |
| ALGO-008: 动态掩模间歇逃逸 | **未解决** | 设计中仍计划使用类似动态掩模机制 |

### 7.4 固有问题与核心矛盾

#### 问题 1：EMA 漂移（作者已知）

作者指出"计算差值有逐渐变大的趋势"。根因：

- **高动量**需求（抗噪声稳定背景） ↔ **低滞后**需求（追踪天空亮度缓变）
- EMA 的遗忘是指数衰减、永不归零，对缓慢系统性变化（天亮/天暗）产生持续偏差
- M3Detector 的滑窗有确定的遗忘窗口（n帧后旧数据完全丢弃），问题相同但可控性更好

#### 问题 2：固定面积比例的三难困境（作者已知）

设计说"确保前景划分为 top-k% 像素"作为阈值基准：

| 场景 | 问题 |
|------|------|
| 无目标帧（占绝大多数时间） | 强制输出 k% 前景 → 全是噪声 → 连通域检测淹没在假区域中 |
| 多/大目标帧 | 目标面积 > k% → 阈值被抬高 → 暗弱目标被截断 |
| 云层运动帧 | 大面积亮度变化消耗 k% 配额 → 真实目标被噪声区域挤出 |

这使得"恒定灵敏度"的设计目标在实际场景中不可能通过单一固定比例实现。

#### 问题 3：运动方向信息丢失

M3Detector 输出线段天然携带方向（起点→终点），Collector 的方向一致性特征 `drst_std` 依赖此信息。连通域检测输出 bounding box 无方向性：

- 需要额外的帧间位移匹配（光流或模板匹配）恢复方向 → 复杂度回升
- 或者改造 Collector 使其不依赖方向特征 → 影响卫星/飞机的辨别能力

#### 问题 4：状态维护与检测耦合

`detect()` 内部调用 `post_update()` 更新 EMA。若未来出现跳帧、异常恢复等绕过 `detect()` 的路径，EMA 将不被更新。应将状态维护显式分离为独立步骤。

### 7.5 如果继续迭代的建议方案

若要将该检测器发展为可用方案，建议以下修改：

| 问题 | 建议 |
|------|------|
| EMA 漂移 | 采用**双 EMA**（快速 + 慢速）：`bg_fast`（momentum=0.9）追踪短期变化，`bg_slow`（momentum=0.999）追踪长期趋势；前景 = `frame - bg_fast`，阈值基于 `bg_fast - bg_slow` 的偏差自适应 |
| 固定面积比例失效 | **放弃面积比例法**，改为"基于 EMA 噪声 std 的倍数"作为阈值（如 `threshold = k * noise_std`），与 M3Detector 的 SNR 方案本质一致但实现更高效 |
| 方向信息缺失 | 连通域检测后，对每个区域计算**主轴方向**（PCA/最小外接矩形角度）；或在连续帧间对区域做质心位移估算 |
| 无目标帧假响应 | 添加**最小面积阈值** + **形态学开运算**过滤小区域噪声；对每帧前景面积做统计，超出正常波动范围才输出 |
| 状态耦合 | 将 `post_update()` 合并到 `update()` 中，用前一帧（非当前帧）更新 EMA：`bg_t = EMA(bg_{t-1}, frame_{t-1})`，`diff_t = frame_t - bg_t` |

### 7.6 总结评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 计算效率潜力 | ★★★★★ | 消除核心性能瓶颈，内存大幅减少 |
| M3Det 缺陷克服 | ★★★☆☆ | 解决性能和丢帧问题，慢速目标部分改善，阈值问题需新方案 |
| 当前可用性 | ★☆☆☆☆ | 原型代码，不可运行 |
| 设计方案可行性 | ★★★☆☆ | 核心思路合理，但面积比例法和方向缺失是硬伤，需要替代方案 |
| 推荐投入 | **中等** | 值得作为下一代检测器开发，但需重新设计阈值策略和方向恢复机制，不建议在当前代码基础上修补 |

---

## 八、算法改进提案

> 以下提案基于对现有代码的完整阅读，旨在为未来迭代提供可选的技术路线评估。

### 8.1 曝光帧数估计（rf_estimator）改进

> 位置：`MetLib/videoloader.py` L629-705（`_rf_est_kernel` + `rf_estimator`）

#### 当前方案

检测前从视频开头/中间/结尾各采样 ~100 帧，计算每帧亮度总和，通过检测亮度序列的"上升沿"间距来推断实际曝光帧数（慢门帧重复模式）。

#### 现有问题

| 问题 | 说明 |
|------|------|
| 上升沿检测噪声敏感 | 4 帧窗口判断局部极值，无幅度阈值过滤，云/星闪烁/飞机等产生大量虚假转折 |
| 硬编码阈值 `0.01` 无效 | `f_sum` 量级为 10^5–10^8，阈值 0.01 仅过滤精确零值 |
| `min(median, clipped_mean)` 系统性低估 | 混入噪声间隔时取最小值偏向低估真实周期 |
| 启动延迟 | 300 帧解码 + 2 次 seek（H.264 非精确 seek 可能更慢），用户可感知 |
| 状态修改无异常保护 | `exp_frame` 修改后若中途异常不会恢复 |

#### 约束条件（设计决策背景）

- **有损压缩**：帧不会精确重复，不能简单用"像素差为零"判断重复帧
- **VFR（可变帧率）**：即使对软件暴露相同 fps，实际曝光可能在视频内变化
- **短片段输入**：上游应用可能将视频分片（~2min）后送入，丢弃前 N 帧结果代价高
- **必须在检测前或检测早期确定**：`exp_frame` 控制 VideoLoader 的帧合并粒度

#### 提议方案：轻量预估 + 在线零成本修正

**核心思路**：用自相关替代上升沿检测（统计性更强），减少预估采样量，检测运行中零成本监控并按需调整。

**Phase 0 — 快速预估（开头 30-50 帧，无 seek）**：

```python
def autocorr_estimate(f_sum: np.ndarray) -> int:
    """基于亮度序列自相关检测周期性帧重复。"""
    # 去线性趋势（对抗缓慢漂移）
    detrended = f_sum - np.linspace(f_sum[0], f_sum[-1], len(f_sum))
    n = len(detrended)
    acf = np.correlate(detrended, detrended, mode='full')[n-1:]
    acf = acf / (acf[0] + 1e-12)
    # 找 lag>=2 的第一个显著峰
    for lag in range(2, min(n // 4, 20)):
        if acf[lag] > acf[lag-1] and acf[lag] > acf[lag+1] and acf[lag] > 0.4:
            return lag
    return 1
```

自相关优势：
- 利用所有帧对的信息（非仅转折点），少量样本即可得到可靠周期估计
- 去趋势后对缓慢亮度漂移免疫
- 搜索空间有限（exp_frame 通常 ≤ 15），无需大量数据
- 启动延迟从 ~300 帧减至 ~50 帧（减少 ~80%），且无 seek

**Phase 1 — 检测中在线监控（零额外 I/O）**：

```python
class ExpFrameMonitor:
    """滑动窗口自相关监控，嵌入检测主循环。"""
    def __init__(self, window_size=150):
        self.window_size = window_size
        self.ring = np.zeros(window_size, dtype=np.float64)
        self.idx = 0
        self.filled = False

    def push(self, frame_sum: float):
        self.ring[self.idx % self.window_size] = frame_sum
        self.idx += 1
        if self.idx >= self.window_size:
            self.filled = True

    def estimate(self) -> int:
        if not self.filled:
            return -1  # 样本不足，维持当前值
        data = np.roll(self.ring, -self.idx % self.window_size)
        data -= np.linspace(data[0], data[-1], len(data))
        acf = np.correlate(data, data, mode='full')[len(data)-1:]
        acf /= acf[0] + 1e-12
        for lag in range(2, min(len(data)//4, 20)):
            if acf[lag] > acf[lag-1] and acf[lag] > acf[lag+1] and acf[lag] > 0.4:
                return lag
        return 1
```

主循环改动（~5 行）：

```python
# 每帧
monitor.push(np.sum(frame))

# 每 ~150 帧
if monitor.filled and frame_count % 150 == 0:
    new_est = monitor.estimate()
    if new_est != video_loader.exp_frame and new_est > 0:
        video_loader.exp_frame = new_est
```

#### 为何可以不丢弃、不重置

`exp_frame` 影响帧合并粒度。若估计偏差：
- **低估（真实 3，用了 1）**：检测器看到未合并单帧，信噪比较低但不是"错的"——噪声响应会被 Collector 运动过滤和 recheck 自然淘汰
- **高估（真实 1，用了 3）**：降低时间分辨率，可能漏极短暂事件——对 2min 短片段影响有限

轻微偏差在整个管线中可容忍，调整后自然收敛。

#### 运行时调整安全性

`ThreadVideoLoader.pop()` 每次调用时动态读取 `self.exp_frame`：

```python
for _ in range(self.exp_frame):  # 每次 pop 时读取
    frame = self.queue.get(...)
```

daemon 线程只负责往 queue 塞单帧，不感知 `exp_frame`。修改该属性在下一次 `pop()` 即刻生效，无需 stop/start/reset。

#### VFR 处理

滑动窗口内的自相关给出**局部估计**而非全局常量。当帧率在视频内变化时，最近 150 帧的自相关峰会移动，监控器自然跟踪。这比当前"三段采样取折衷统计量"更直接地应对 VFR。

#### 对比总结

| | 当前方案 | 提议方案 |
|---|---|---|
| 启动延迟 | 300 帧 + 2 次 seek | 30-50 帧，无 seek |
| VFR 支持 | 三段折衷统计 | 滑动窗口在线追踪 |
| 丢弃结果 | 否 | 否 |
| 算法鲁棒性 | 上升沿检测，噪声敏感 | 自相关，统计性更强 |
| 额外 I/O | 纯预估帧与检测分离 | 预估 50 帧 + 检测中零成本 |
| 实现改动 | — | 替换 `rf_estimator` + 主循环加 `monitor.push()` |

#### 待细化

- `acf > 0.4` 的峰值阈值需实验标定（过低误检噪声周期，过高漏检弱周期）
- 在线调整后是否需要通知 Detector（若 Detector 内部有基于 exp_frame 的时间换算）
- 是否需要对首次预估给一个"置信度"，低置信度时更频繁地检查

---

### 8.2 第一阶段检测器：候选选取改进

#### 当前方案核心逻辑回顾

M3Detector 的 `max - mean` 思路本身合理：流星出现在窗口 1-3 帧中，max 捕获峰值，mean 将其稀释为 `brightness/n`，差值约为 `brightness × (n-1)/n`。**问题不在思路，在实现方式和鲁棒性。**

#### 提案 A：帧间正差分累积（推荐优先尝试）

**核心计算**：

```
d[t] = clip(frame[t] - frame[t-1], min=0)     # 仅保留"变亮"方向的差分
accum = max(d[t], d[t-1], ..., d[t-n+1])      # 窗口内最大正差分
```

**优势**：

| 特性 | 效果 |
|------|------|
| 静态天体（星点）帧间差为 0 | **天然过滤**，无需动态掩模 |
| 增量维护：新帧进入时 `accum = max(accum, d_new)`，帧过期时才需重算 | **摊销 O(H×W)**，绝大多数帧无需 refresh |
| 慢速目标每帧有小量正差分 | 累积后**可检出**（M3Det 中 max≈mean 不可检） |
| 突发大面积事件（云、抖动）所有像素同时变化 | 可通过"正差分面积比例 > 阈值则跳过"优雅处理，无需硬编码 500 线上限 |

**退化处理**：帧过期时若恰好是当前 max 的来源帧，才需重算。对 uint8 图像可用 256-bin 直方图维护——max = 最高非零 bin，更新 O(1)。实际中流星亮度远高于噪声，max 通常在窗口末端（最近帧），过期触发率极低。

**与现有架构的兼容性**：累积差分图上仍可接 Hough 直线检测（保留方向信息），Collector 无需改动。由于噪声被帧间差大幅抑制，二值化阈值可更低、Hough 输出更干净。

#### 提案 B：差分图 + 连通域 + 主轴方向（替代 Hough）

在提案 A 的 `accum` 图上：

1. 自适应阈值：基于 `accum` 全图 std 的 k 倍（无需子区域采样，解决 ALGO-005）
2. 形态学闭运算 → `cv2.connectedComponentsWithStats`
3. 对每个连通域：`cv2.minAreaRect` 获取最小外接旋转矩形 → 长轴为运动方向，长宽比区分线型/面积型

彻底消除 Hough 的不稳定性（ALGO-003），连通域计算量稳定可控。

**方向信息恢复**：`minAreaRect` 返回的角度即为主轴方向，对于流星轨迹（长条形连通域）精度足够。对于面积型检测（如闪电），方向本身无意义，标记为面积类型即可。

#### 提案 C：双时间尺度差分（解决慢速目标）

针对 ALGO-001（慢速目标在短窗口内差分不足），引入第二时间尺度：

```
short_diff = max_over_1s(frame_diffs)     # 快速响应，检测流星
long_diff  = max_over_5s(frame_diffs)     # 慢速响应，检测卫星/飞机
```

两个尺度独立检测、输出合并后由 Collector 分类。慢速目标不再被短窗口淹没——在长窗口中积累了足够帧间差分。内存开销仅增加一份 `accum` 缓冲区。

#### 三方案对比

| | 提案 A (帧差累积) | 提案 B (+连通域) | 提案 C (+双尺度) |
|---|---|---|---|
| 改动复杂度 | 低，替换 diff 计算即可 | 中，替换 Hough 为连通域 | 中高，双通路架构 |
| 解决 ALGO-001 慢速目标 | 部分改善 | 部分改善 | **完全解决** |
| 解决 ALGO-002 性能瓶颈 | **完全解决** | **完全解决** | 完全解决 |
| 解决 ALGO-003 丢帧 | **完全解决** | **完全解决** | 完全解决 |
| 方向信息保留 | 保留（仍用 Hough） | 通过主轴恢复 | 保留 |
| 与现有 Collector 兼容 | **无缝** | 需适配区域输出格式 | 需适配双来源合并 |

**推荐路径**：A → B → C 递进实施。A 为最小改动验证核心思路；B 解决 Hough 稳定性；C 解决慢速目标的最后一块短板。

---

### 8.3 第二阶段 Collector：特征与分类器改进

#### 当前特征集诊断

| 现有特征 | 区分目标 | 局限性 |
|----------|---------|--------|
| `fix_speed` | 流星 vs 卫星 | 焦距不同时归一化后范围差异大 |
| `fix_duration` | 流星 vs 飞机 | 与 fps 强相关，低 fps 时分辨力下降 |
| `fix_dist` | 流星 vs 噪声 | 对短暗流星不友好 |
| `drst_std` | 线性 vs 飞虫 | 仅 3 帧样本时 std 估计极不稳定 |

**根本问题**：4 个特征 × 梯形函数 × 乘法 = 信息维度不足 + 组合方式脆弱。

#### 建议新增特征

**运动学特征**：

| 特征 | 计算方式 | 鉴别价值 |
|------|---------|---------|
| 加速度 `acceleration` | 相邻帧对速度差的均值 | 流星减速 / 卫星恒速 / 飞虫变速 |
| 速度方差系数 `speed_cv` | std(帧间速度) / mean(帧间速度) | 匀速≈0，飞虫>>0 |
| 直线拟合残差 `linearity` | 观测点到最小二乘线的平均距离 / 轨迹长度 | 流星/卫星<0.02，飞虫/云>0.1 |

**光度学特征**：

| 特征 | 计算方式 | 鉴别价值 |
|------|---------|---------|
| 亮度偏度 `brightness_skew` | 轨迹各帧亮度序列的 skewness | 流星"快亮慢灭"负偏度，卫星对称 |
| 亮度衰减率 `fade_rate` | (max_brightness - end_brightness) / duration | 流星高，卫星≈0 |
| 峰值位置 `peak_position` | argmax(brightness) / total_frames | 流星偏前（<0.3），卫星居中（≈0.5） |

**形态学特征**：

| 特征 | 计算方式 | 鉴别价值 |
|------|---------|---------|
| 宽度一致性 `width_cv` | std(各帧检测宽度) / mean | 流星一致，云/虫不规则 |
| 长宽比 `aspect_ratio` | 叠加图最小外接矩形 长/宽 | 流星>5，精灵<3，闪电不规则 |

以上特征均可在现有 `MeteorSeries` 维护的逐帧观测数据上计算，无需额外数据源。

#### 分类器升级方案

**方案 1：加权对数几率（最小改动，解决 ALGO-013）**

将 `P = P1 × P2 × P3 × P4` 改为：

```python
log_odds = w1*log(P1) + w2*log(P2) + w3*log(P3) + w4*log(P4) + bias
P_final = sigmoid(log_odds)
```

本质是手动形式的逻辑回归。权重可学习或手动设定（如 `w_drct=0.5` 降低方向因子杀伤力）。**单一因子偏离不再致命**，与现有梯形函数完全兼容。

**方案 2：轻量级梯度提升树（推荐，需标注数据）**

```python
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=50, max_depth=4, num_leaves=8)
model.fit(features, labels)  # features: N×10+, labels: {meteor, satellite, bug, noise, ...}
```

**为何特别适合该场景**：

| 优势 | 说明 |
|------|------|
| 天然处理特征交互 | 学到"慢速+短时长=暗弱流星" vs "慢速+长时长=卫星"，乘法模型无法表达 |
| 对异常值鲁棒 | 单特征 NaN 不崩溃（树走另一分支），彻底消除 ALGO-013 |
| 可解释 | `feature_importance` 直接指示哪些特征有效 |
| 推理极快 | 50 棵 4 层树推理 <0.1ms，远小于 YOLO recheck |
| 训练数据已有 | `evaluate.py` 标注格式 + 检测结果 = 天然训练集 |

**部署方式**：导出为 ONNX 或展开为纯 Python if-else（50 棵浅树可直接编码），无额外运行时依赖。

**方案 3：级联结构（远期优化）**

```
阶段1: 快速规则过滤 → 排除明显噪声（面积<min, 持续<1帧），O(1)
阶段2: 特征分类器（LightGBM）→ 精确分类，<1ms
阶段3: YOLO recheck → 仅对阶段2置信度 ∈ [0.3, 0.7] 的样本调用
```

效果：阶段 1 过滤 ~80% 噪声；阶段 3 调用量减少 70%+（目前所有通过前置检查的样本都送 YOLO）。总计算开销显著降低，同时保持甚至提升准确率。

#### 分类器改进优先级

| 改进项 | 难度 | 收益 | 建议顺序 |
|--------|------|------|---------|
| 加权对数几率替换乘法模型 | 低 | 暗弱目标召回提升，消除 ALGO-013 | **1** |
| 新增 3-4 个运动学+光度学特征 | 中 | 卫星/飞机区分度显著提升 | **2** |
| LightGBM 分类器替换梯形函数 | 中 | 整体准确率跃升 | **3**（需积累标注数据） |
| 级联结构优化 recheck 调用 | 中 | 降低计算开销 50%+ | **4** |

---

### 8.4 综合实施路线图

```
Phase 1 (短期，1-2周)
├── 检测器：帧差累积替换 max-mean（提案 A）
├── 分类器：加权对数几率替换乘法（方案 1）
└── 验证：evaluate.py 对比基线

Phase 2 (中期，2-4周)
├── 检测器：连通域替换 Hough（提案 B）
├── 分类器：新增运动学+光度学特征
├── 数据：积累标注样本用于训练
└── 验证：多视频源回归测试

Phase 3 (远期，1-2月)
├── 检测器：双时间尺度（提案 C）
├── 分类器：LightGBM 替换手工规则（方案 2）
├── 架构：级联 recheck 结构（方案 3）
└── 验证：全面性能+效果评估
```

各阶段独立可交付，每阶段结束后可通过 `evaluate.py` 量化评估是否达到预期收益再决定是否继续。

---

## 九、工程基础设施改进

### 9.1 模块级测试（最高优先）

**为何排最高**：这是所有其他改进的安全网。当前"无测试 + 靠 evaluate.py 跑完整视频验证"意味着：
- 每次修改只能靠端到端结果判断对错，定位问题极慢
- 架构审查中的多个 bug（ARCH-001, ALGO-025 等）之所以存活多版本，正是因为没有边界情况覆盖
- 后续算法替换需要对比实验，没有测试基础设施无法快速迭代

**建议范围**（覆盖"改动时会炸的边界"，不追求覆盖率）：

| 模块 | 测试要点 |
|------|---------|
| `utils.py` | SlidingWindow 边界（n=1, 满窗, 溢出）、EMA warmup 数值收敛、除零 guard |
| `model.py` | 单帧推理输入输出形状、多尺度 tile 覆盖完整性、NMS 输出格式 |
| `collector.py` | 轨迹关联逻辑（无匹配/首匹配/边界距离）、prob_meteor 边界值、NaN 输入 |
| `videoloader.py` | 曝光估算对各种 fps 的行为、ThreadVideoLoader start/stop 生命周期 |
| `metstruct.py` | dacite 反序列化各 config 变体（M3Det/DLDet/Brightness）、to_dict 继承字段 |

**工具选择**：`pytest`，秒级完成，与 evaluate.py 的分钟级端到端测试互补。

**与计划表关系**：应在"修正确性 Bug"（六-1）**之前或同时**建立，确保修 bug 时有回归保护。

**测试层级设计**：

```
单元测试 (pytest, <10s)          ← 本项新增
├── utils/metstruct/model 边界
├── collector 逻辑
└── videoloader 生命周期

集成测试 (evaluate.py, ~分钟级)  ← 已有
├── 端到端效果回归 (P/R/F1)
└── 性能基线 (时间/CPU/内存)
```

---

### 9.2 主接口 API 化（中等优先）

**当前阻碍被作为 package 调用的问题**：

| 问题 | 影响 |
|------|------|
| 全局状态（`set_resource_dir`, `set_default_logger`, `_ensure_class_names_loaded`） | 调用方必须按特定顺序初始化，不能并发多实例 |
| `exit()` 在库代码中（ARCH-002） | 作为 package 调用时直接杀死宿主进程 |
| 日志直接写 stdout | 嵌入其他系统时污染输出流 |
| 配置只能从 JSON 文件加载 | 不能 programmatically 构造配置对象 |
| 无 `__init__.py` 在项目根 | `import metdetpy` 不可用 |

**建议 API 设计**：

```python
# 目标用法
from metdetpy import detect_video, detect_photo
from metdetpy.config import M3DetConfig, DLDetConfig

cfg = M3DetConfig(resize=960, sensitivity="normal")
result = detect_video("video.mp4", config=cfg)  # 返回 MDRF dataclass
```

**前置依赖**：
- ARCH-002（去除 exit）
- ARCH-006（全局状态改为 context/参数传递）
- ARCH-008（异常处理统一）

这些依赖恰好是重构路径 Phase 1-2 的产出，API 化是 Phase 2 完成后的自然结果，额外工作量仅为添加根 `__init__.py` + 稳定化公开接口签名。

---

### 9.3 模型推理硬件泛用性（最低优先）

**当前痛点**：

| 平台 | 当前方案 | 问题 |
|------|---------|------|
| Windows | `onnxruntime-directml` | 与 `onnxruntime-gpu` wheel 互斥，打包时必须二选一 |
| macOS | `onnxruntime`（内含 CoreML EP） | 独立构建，与 Windows 不同 wheel |
| Linux | `onnxruntime` / `onnxruntime-gpu` | CUDA 版本绑定 |

**根因**：onnxruntime 各 Execution Provider 的 wheel 互斥是 PyPI 分发限制，非本项目能解决。

**建议方案**：

| 时间范围 | 方案 |
|---------|------|
| 短期 | 在 `make_package.py` 中根据目标平台自动选择依赖；`model.py` 中已有的运行时 provider 检测逻辑保持不变 |
| 中期 | 统一到 `onnxruntime >= 1.17`，利用其改进的 provider fallback 机制；添加安装时诊断脚本（检测可用 EP 并建议安装） |
| 远期 | 如需 TensorRT 支持，引入模型格式转换管线（ONNX → TensorRT engine），或等待 onnxruntime TensorRT EP 稳定 |

**与计划表关系**：完全独立于功能开发，不阻塞任何改进。可以在任何时候做。优先级低于测试和 API 化是因为：当前方案**能跑**（只是管理复杂），而测试缺失和 API 缺失**阻碍开发效率**。

---

### 9.4 综合优先级总览

```
时间线        架构/算法改进                    工程基础设施
─────────────────────────────────────────────────────────────
Phase 0      修正确性 Bug ◄────────────────── 模块级测试 (同步建立)
             │
Phase 1      数值健壮性 / config dispatch
             │
Phase 2      VideoLoader重构 / 错误处理 ───► API化 (作为自然产出)
             │
Phase 3      算法改进（检测器+分类器）
             │
任意时机      ─────────────────────────────── 推理硬件泛用性 (独立)
```

**关键路径**：模块级测试 → Bug 修复 → 架构重构 → API 化 → 算法改进。推理硬件优化不在关键路径上，可并行处理。


--------

## 已经修复或者优化的缺陷

### ~~TODO-ARCH-001: ProcessVideoLoader 致命 Bug（P0）~~

**状态**: 已移除  
**处理方式**: `ProcessVideoLoader` 已从代码库中移除。视频解码后端（OpenCV/PyAV）均为释放 GIL 的 C 扩展调用，线程级 `ThreadVideoLoader` 已足够实现并行解码，进程级隔离无实际性能优势且引入显著 IPC 开销。

---

### ~~TODO-ARCH-002: Collector 硬杀进程（P0）~~

**状态**: 已修复  
**处理方式**: `prob_meteor` 检测到 NaN 时不再调用 `exit()`，改为返回 `0.0`（最低置信度），使该 meteor 自然走入 drop 路径被丢弃。NaN 是单条轨迹的数据污染问题，不影响其他追踪序列，无需终止整个检测流程。

---

### ~~TODO-ALGO-018: 多尺度双重 NMS 语义不一致~~

**状态**: 已修复

**处理方式**: 单 tile 与跨 tile NMS 统一使用 class-aware 实现；单 tile 直接将 YOLO 的中心点 `cxcywh` 转为 OpenCV 左上角 `tlwh`，NMS 后再将保留框转为 `xyxy`，跨 tile 则由 `xyxy` 转为 `tlwh`。两级 NMS 均直接以未开方的 `objectness × class` 联合分数执行候选过滤与排序，并使用同一个 `pos_thre`；`sqrt` 仅在最终 API 返回时执行，不影响候选集合或 NMS。跨 tile 合并复用可配置的 `nms_thre`，不再硬编码 IoU=0.1。联合分数阈值将在静态图像回归测试后重新标定。

---
