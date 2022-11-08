# 参数设置说明

## General

```json
"resize_param": 960,
"exp_time": "auto"
```

|参数名|格式|说明|
|------|---|---|
|resize_param| int,array|指定检测使用的分辨率。较低的分辨率下程序运行更快，较高的分辨率则有助于检测到更暗弱和短的流星。默认的960适用于绝大多数情况。|
|exp_time|float, str("auto", "slow", "real-time")|单帧的曝光时间|

## Stacker

```json
"stacker": "MergeStacker",
"stacker_cfg": {
    "pfunc": "max"
}
```

> stacker: 描述使用的帧如何被传递进入检测器。
> stacker_cfg: 描述检测器的配置。
>> pfunc: 当使用"MergeStacker"时起效，描述多帧通过何种算法合并为一帧传递给检测器。

## Detector

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

## Collector

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

meteor_cfg = dict(
    min_len=10,
    max_interval=4,
    time_range=(0.12, 10),
    speed_range=(1.6, 4.6),
    thre2=320)
