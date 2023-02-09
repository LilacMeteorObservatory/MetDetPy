# 参数设置说明

## 预处理/Preprocessing

涉及到读入和预处理的参数于此处设置。

```json
"resize_param": 960,
"exp_time": "auto"
```

|参数名|格式|说明|
|------|---|---|
|resize_param| int,array|指定检测使用的分辨率。较低的分辨率下程序运行更快，较高的分辨率则有助于检测到更暗弱和短的流星。可以同时指定宽高（如`[960, 540]`）或仅指定长边（如`960`）。推荐仅指定长边，其可以自适应任意长宽比。预设的960适用于绝大多数情况。|
|exp_time|{float, str(`"auto"`, `"slow"`, `"real-time"`)}|单帧的曝光时间。可以使用浮点数指定该时间。如果实际曝光时间与帧率不匹配（例如，部分相机可以输出曝光时间为1/20s的4k60p的视频，此时每连续3帧会共用相同图像），指定曝光时间可以有效改善运行速度和识别的准确率。如果确定帧率与曝光时间匹配，则可以使用`"real-time"`。此外，可以使用`"auto"`，此时程序会自动根据片段估算实际帧率（会在启动前花费一小段时间进行估算）。|


⚠️ 有关选项将在`v2.0.0`中正式合并入`"preprocessing"`项中。为保持兼容性，`v2.0.0`前两种写法均可使用。`"preprocessing"`的示例如下：

```json
"preprocessing":{
    "resize_param": 960,
    "exp_time": "auto",
    "merge_func": "max"
}
```
其中新增参数说明如下：
|参数名|格式|说明|
|------|---|---|
|merge_func|str|描述多帧图像通过何种算法合并为一帧传递给检测器。可选`"max"`,`"m3func"`及`"mix"`（待实现）。为保持兼容性，`v2.0.0`前该项并非必须包含，但该项和Stacker相关项至少需要在配置文件中提供一个。|

## 堆栈器/Stacker

涉及到读入后帧合并的处理于此处设置。

⚠️ 由于`"SimpleStacker"`实质上是`"MergeStacker"`的特殊情况，有关选项计划将在`v2.0.0`废弃。可以通过在`exp_time`项指定`"real-time"`实现`SimpleStacker`的效果。`pfunc`则将合并到[Preprocessing](#预处理preprocessing)中，并更名为`merge_func`。为保持兼容性，`v2.0.0`前两种写法均可使用。


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

设置使用何种检测器，以及使用的检测超参数。

```json
"detector": "M3Detector",
"detect_cfg": {
    "window_sec": 0.36,
    "bi_threshold": 5,
    "median_sampling_num": -1,
    "line_threshold": 10,
    "line_minlen": 16
}
```

> detector:
> detector_cfg:
>> windows_sec: 
>> bi_threshold ：描述检出流星所使用的阈值。可以根据使用的ISO进行调整，过低可能会引入更多噪声误检。
>> median_sampling_num ：描述中位数的采样数目。更少的采样数目可能会引发低信噪比导致的误差，但可以达到更高的速度。设置-1时表示不跳采。
>> line_* : 直线检测参数。默认情况下不用动。

## 收集器/Collector

```json
"meteor_cfg": {
        "min_len": 10,
        "max_interval": 2,
        "time_range": [
            0.12,
            10
        ],
        "speed_range": [
            1,
            5
        ],
        "thre2": 512
    }
```

> meteor_cfg
>> min_len ：开始记录流星所需要的最小长度（占长边的比）。
>> max_interval：流星最长间隔时间（经过该时间长度没有响应的流星将被认为已经结束）。单位：s。
>> time_range ： 描述流星的持续时间范围。超过或者没有到达阈值的响应将会被排除。单位：s。
>> speed_range ： 描述流星的速度范围。超过或者没有到达阈值的响应将会被排除。单位：frame^(-1)。
>> thre2 ： 描述若干响应之间允许的最长距离平方。