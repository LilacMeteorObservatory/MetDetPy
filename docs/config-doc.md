# 参数设置说明

## 预处理/Preprocessing

涉及到读入和预处理的参数于此处设置。示例配置如下：

```json
"preprocessing":{
    "resize_param": 960,
    "exp_time": "auto",
    "merge_func": "max"
}
```

|参数名|格式|说明|
|------|---|---|
|resize_param| int,array|指定检测使用的分辨率。较低的分辨率下程序运行更快，较高的分辨率则有助于检测到更暗弱和短的流星。可以同时指定宽高（如`960:540`或者`960x540`）或仅指定长边（如`960`）。推荐仅指定长边（适用于不同长宽比视频，无需再调整）。预设的960适用于绝大多数情况。|
|exp_time|{float, str(`"auto"`, `"slow"`, `"real-time"`)}|单帧的曝光时间。可以使用浮点数指定该时间。如果实际曝光时间与帧率不匹配（例如，部分相机可以输出曝光时间为1/20s的4k60p的视频，此时每连续3帧会共用相同图像），指定曝光时间可以有效改善运行速度和识别的准确率。如果确定帧率与曝光时间匹配，则可以使用`"real-time"`。此外，可以使用`"auto"`，此时程序会自动根据片段估算实际帧率（会在启动前花费一小段时间进行估算）。|
|merge_func|str|描述多帧图像通过何种算法合并为一帧传递给检测器。可选`"max"`,`"m3func"`及`"mix"`（待实现）。为保持兼容性，`v2.0.0`前该项并非必须包含，但该项和Stacker相关项至少需要在配置文件中提供一个。|


⚠️ 以下是`v1.2.3`以前所使用的参数用法，将在`v2.0.0`后正式废弃。使用该写法时，需要在`"stacker_cfg"`中配置`"pfunc"`选项。为保持兼容性，`v2.0.0`前两种写法均可使用。

```json
"resize_param": 960,
"exp_time": "auto"
```

## 堆栈器/Stacker

涉及到读入后帧合并的处理于此处设置。

⚠️ 由于`"SimpleStacker"`实质上是`"MergeStacker"`的特殊情况，有关选项将在`v2.0.0`废弃。可以通过在`exp_time`项指定`"real-time"`实现`SimpleStacker`的效果。`pfunc`则将合并到[Preprocessing](#预处理preprocessing)中，并更名为`merge_func`。为保持兼容性，`v2.0.0`前两种写法均可使用。


```json
"stacker": "MergeStacker",
"stacker_cfg": {
    "pfunc": "max"
}
```

|参数名|格式|说明|
|------|---|---|
|stacker|str|描述使用的帧如何被传递进入检测器。可选`"SimpleStacker"`与`"MergeStacker"`。|
|stacker_cfg|json|描述检测器的配置。该选项计划调整。|
|pfunc|str|描述多帧通过何种算法合并为一帧传递给检测器。仅当"MergeStacker"时起效。可选`"max"`,`"m3func"`及`"mix"`。|

## 检测器/Detector

设置使用何种检测器，以及使用的检测超参数（包括二值化阈值算法与直线检测的相关设置）。

```json
"detector": "M3Detector",
"detect_cfg": {
        "window_sec": 1,
        "adaptive_bi_thre": true,
        "bi_cfg": {
            "init_value": 5,
            "sensitivity": "normal",
            "area": 0.1
        },
        "hough_threshold": 10,
        "min_len": 10,
        "max_gap": 10
    },
```

|参数名|格式|说明|
|------|---|---|
|detector|str|描述使用的检测器。|
|windows_sec|int,float|滑动时间窗长度。|
|adaptive_bi_thre|bool|是否启用自适应二值化阈值算法。|
|init_value|int|初始化的二值化阈值。当不启用自适应二值化阈值算法时，其直接描述检出流星所使用的二值化阈值。|
|sensitivity|str|描述自适应二值化阈值算法的灵敏度。|
|area|float|描述自适应二值化阈值算法的采样区域大小。|
|hough_threshold|int|hough直线检测的阈值。|
|min_len|int|描述构成直线的最小长度。|
|max_gap|int|描述构成直线允许的最大间隔。|

⚠️ 有关直线检测的选项计划将在`v2.0.0`调整到`"detect_cfg"`下的`"line_cfg"`中，即：

```json
"detector": "M3Detector",
"detect_cfg": {
        "window_sec": 1,
        "adaptive_bi_thre": true,
        "bi_cfg": {
            "init_value": 5,
            "sensitivity": "normal",
            "area": 0.1
        },
        "line_cfg": {
            "hough_threshold": 10,
            "min_len": 10,
            "max_gap": 10
        }
    },
```

但目前该调整仅在计划中。
## 收集器/Collector

设置流星的过滤和收集条件。

```json
"meteor_cfg": {
        "min_len": 10,
        "max_interval": 3,
        "time_range": [
            0.25,
            6
        ],
        "speed_range": [
            0.9,
            12
        ],
        "drct_range": [
            0,
            0.17
        ],
        "pos_threshold": 0.5,
        "thre2": 512
    }
```

|参数名|格式|说明|
|------|---|---|
|min_len|int|开始记录流星所需要的最小长度(px)。|
|max_interval|int|流星间的最长间隔时间（经过该时间长度没有额外响应，这一个/组流星将被认为已经结束）。单位：s。|
|time_range|array|描述流星的持续时间范围。超过或者没有到达阈值的响应将会被排除。单位：s。|
|speed_range|array|描述流星的允许速度范围。超过或者没有到达阈值的响应将会被排除。单位：$f^{-1}$。|
|drct_range|array|描述流星的直线程度范围。越接近0，该流星越接近理想直线。|
|pos_threshold|float|描述正样本流星的阈值，超过该得分的流星被认为是正样本流星。取值为$\mathbb{R} \in [0,1]$。 |
|thre2|int|描述若干响应之间允许的最长距离平方。|

⚠️ 有关过滤阈值的设计均以running resolution为基准，计划会在`v2.0.0`调整。