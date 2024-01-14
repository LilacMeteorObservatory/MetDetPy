# Configuration Document

<center>Language: English | <a href="./config-doc-cn.md">简体中文</a>  </center>

## Introduction

MetDetPy reads arguments from configuration files. For most circumstances, preset configuration files work well, but there are also times when better detection results can be achieved by adjusting detection arguments. This document explains the meanings of arguments so that they can adjusted according to the requirement.

A configuration file is a `JSON` format text that includes 4 parts: [description](#description), [loader](#loader), [detector](#detector), and [Collector](#collector). As a reference, predefined settings are stored under the [config](../config) directory. It is recommended to take [the default configuration file](../config/m3det_normal.json) as an instance while reading this instruction to better understand the format of the configuration file.

⚠️ APIs in this document may still be changed until the formal `v2.0.0` is released.


## Description

The `description` part describes the name of the config and a simple introduction to it. This is designed for the frontend applications(since most times MetDetPy serves as the backend) and is not essential for detection. 

An example of the `description` is as follows:

```json
"description": {
    "preset_name": "configuration_name",
    "intro": "a_brief_introduction",
}
```

Also, if you wish the content to be different under different language settings, you can fill the `description` as follows:

```json
"description": {
    "preset_name_cn": "M3Det-通用",
    "preset_name_en": "M3Det-general",
    "intro_cn": "自适应检测器，适用于多数检测场景。",
    "intro_en": "Adaptive meteor detector. Suitable for most detection scenarios."
}
```

## Loader

The `loader` part mainly manages arguments about video loading and frame preprocessing. An example is as follows:

```json
"loader": {
    "name": "ThreadVideoLoader",
    "wrapper": "OpenCVVideoWrapper",
    "resize": 960,
    "exp_time": "auto",
    "merge_func": "max",
    "grayscale": true
}
```

The instructions of these arguments is as follows:


|Argument|Type|Explanation|Recommendation|
|------|---|---|--------|
|name|str|Name of the `loader`. Should be chosen from `"VanillaVideoLoader"`,`"ThreadVideoLoader"`, and `ProcessVideoLoader`. Generally, it is recommended to use `"ThreadVideoLoader"`, because it is well-tested, stable, and fast.|`"ThreadVideoLoader"`|
|wrapper|str| Name of the `wrapper` that is used to load the video. Now only the `"OpenCVVideoWrapper"` based on OpenCV is supported. |`"OpenCVVideoWrapper"`|
|resize|int, array, str| The resolution that is used during the detection. A lower resolution makes detection faster, while a higher one helps detect shorter meteors. It is recommended to input an integer to specify the length of the long side because the program will scale the input video automatically. It is also supported to specify the height and the width of the video at the same time(by inputting a string consisting of two numbers split with `:` or `x`, like `"960:540"`，`"960x540"`; also a list is also supported, like `[960,540]`) |`960`|
|exp_time|float, str(`"auto"`, `"slow"`, `"real-time"`)| The exposure time of a single frame. When using traditional detectors, it is recommended to set `"auto"`, since MetDetPy will estimate the real exposure time according to the video. If the exposure time in the meta-data matches the real one, set `"real-time"`. To specify a value, input a float in second. |`"auto"` (LineDetector)/ `0.5` (MLDetector)|
|merge_func|str| A function name that indicates how multiple frames are merged. Should be chosen from `"max"`,`"m3func"`,`"mix_max_median_stacker"`. It is recommended to use `"max"` due to the visual characteristic of meteors.|`"max"`|
|grayscale|bool| Whether to load the video in grayscale. For LineDetector, `grayscale` must be set to `true`; for deep learning-based detector, it should be `false`. |`true`(LineDetector)/ `false`(MLDetector)|

⚠️ 
1.  `"name": "ProcessVideoLoader"` is an experimental feature, and is still not fully tested. It cannot be used on MacOS and is not recommended to be used in the production environment.
2. Q: Why the exposure time is needed? A: There are conditions when the real exposure time and frame per second (FPS) do not match. (for example, some cameras encode 4k60p videos with an exposure time of 1/20s. In these videos, frames change every 3 ticks instead of 1.) Using real exposure time would improve accuracy and efficiency.

## Detector

The `detector` manages the name of the detector, the length of the detection sliding-window, and other detector-related arguments (like binary threshold and line detection settings).

An example for line detector is as follows:

```json
"detector": {
    "name": "M3Detector",
    "window_sec": 1,
    "cfg": {
        "binary": {
            "adaptive_bi_thre": true,
            "init_value": 7,
            "sensitivity": "normal",
            "area": 0.1,
            "interval": 2
        },
        "hough_line": {
            "threshold": 10,
            "min_len": 10,
            "max_gap": 10
        },
        "dynamic": {
            "dy_mask": true,
            "dy_gap": 0.05,
            "fill_thre": 0.6
        }
    }
}
```

And an example for deep learning-based detector is as follows:

```json
 "detector": {
    "name": "MLDetector",
    "window_sec": 1,
    "cfg": {
        "model": {
            "name":"YOLOModel",
            "weight_path": "./weights/yolov5s.onnx",
            "dtype": "float32",
            "nms": true,
            "warmup": true,
            "pos_thre": 0.25,
            "nms_thre": 0.45
        }
    }
}
```

The instructions of these arguments is as follows:


|Argument|Type|Explanation|Recommendation|
|------|---|---|---|
|name|str|Name of the detector. Should be chosen from `"ClassicDetector"`, `"M3Detector"`, and `"MLDetector"`. `"ClassicDetector"` and `"M3Detector"` are traditional detectors based on line detection, while `"MLDetector"` is a deep learning-based detector. Basically, traditional detectors are more sensitive and faster, which may cause more false positive samples (this can be improved by configuring [recheck](#recheck_cfg)); meanwhile, deep learning-based detectors are more robust, with higher computing cost. These two types of detectors also differ in their arguments.|`"M3Detector"`(traditional detector)/ `"MLDetector"` (deep learning detector)|
|windows_sec|int,float| Describes the time length of the sliding window. Meteors are detected within the sliding window when running. Generally it is recommended to set this around `1` second. |`1`|
|cfg|-|Describes the specific arguments of the detector. |(See below)|

When using traditional detectors, it is required to config `"binary"`(binary threshold arguments), `"hough_line"`(line detection arguments) and `"dynamic"`(dynamic mechanism arguments) under the `cfg` of the `detector`. When using deep learning-based detectors, it is required to config `"model"` under the `cfg` of the `detector`. Specific requirements of these arguments are as follows:


<table>
    <tr>
        <td><b>Belonging</b></td> 
        <td><b>Name</b></td> 
        <td><b>Type</b></td> 
        <td><b>Explanation</b></td> 
        <td><b>Recommendation</b></td> 
   </tr>
    <tr>
        <td rowspan="5"><p>binary</p>(binary threshold arguments)</td>    
        <td>adaptive_bi_thre</td> 
        <td>bool</td> 
        <td>Apply adaptive binary threshold algorithm. This algorithm can estimate best binary threshold according to the signal noise rate (SNR) of the sampling area. (Binary threshold is important for line detect-based detector.)</td> 
        <td>true</td> 
    </tr>
    <tr>
        <td>init_value</td>  
        <td>int</td> 
        <td>Initial binary threshold. When adaptive binary threshold is not applied, this value will be used as the constant binary threshold.</td> 
        <td>5</td>  
    </tr>
    <tr>
        <td>sensitivity</td>  
        <td>str</td> 
        <td>Sensitivity of the adaptive binary threshold algorithm. Different sensitivity function will map the SNR value to the corresponding threshold. When adaptive binary threshold is not applied, sensitivity will not work. </td> 
        <td>normal</td>  
    </tr>
    <tr>
        <td>area</td>  
        <td>float</td> 
        <td>Sampling area size of the adaptive binary threshold algorithm. Ranged [0,1], representing the area ratio of the whole image. To set a proper value will accelerate estimation while getting approrixate value. It is recommended to set between 0.05 to 2. Illegal value (>1 or <0) setting will lead to use the whole area to estimate. </td> 
        <td>0.1</td>  
    </tr>
    <tr>
        <td>interval</td>  
        <td>int</td> 
        <td>设置阈值估算间隔。过于频繁的阈值估算会增大计算量，降低程序运行性能；过低的计算频率则可能无法及时根据信噪比变化变。</td> 
        <td>2</td>  
    </tr>
    <tr>
        <td rowspan="3"><p>hough_line</p>(Hough直线检测相关参数。参数具体含义参考<a href="https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html">OpenCV官方文档</a>)</td>  
        <td>hough_threshold</td> 
        <td>int</td>  
        <td>hough直线检测的阈值。</td>
        <td>10</td>  
    </tr>
    <tr>
        <td>min_len</td>  
        <td>int</td> 
        <td>描述构成直线的最小长度。</td> 
        <td>10</td>  
    </tr>
    <tr>
        <td>max_gap</td>  
        <td>int</td> 
        <td>描述直线间允许的最大间隔。</td> 
        <td>10</td>  
    </tr>
    <tr>
        <td rowspan="3"><p>dynamic</p>(动态机制相关参数)</td>  
        <td>dy_mask</td> 
        <td>bool</td>  
        <td>是否启用动态掩模机制。动态掩模对持续产生响应的区域施加掩模，能够降低在亮星点和持续干扰附近区域的响应。</td>
        <td>true</td>  
    </tr>
    <tr>
        <td>dy_gap</td>  
        <td>float</td> 
        <td>动态间隔参数。该项使hough参数中的"max_gap"在响应过多时减少，提高对直线的要求，减少二值化计算偏差时的误报。该值表示随着潜在流星区域面积增大到给定值时，"max_gap"衰减到0。如配置0.05，则潜在流星区域面积在大于0.05%时"max_gap"衰减到0。</td> 
        <td>0.05</td>  
    </tr>
    <tr>
        <td>fill_thre</td>  
        <td>float</td> 
        <td>描述构成直线允许的最大中空比例，用于降低非连续直线的误报。设置为0时表示不生效。</td> 
        <td>0.6</td>  
    </tr>
    <tr>
        <td><p>model</p>(模型相关参数)</td>  
        <td>-</td>  
        <td>-</td> 
        <td>参考<a href=#模型model>模型</a>部分。</td> 
        <td>0.6</td>  
    </tr>
</table>