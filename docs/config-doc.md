# Configuration Document

<center>Language: English | <a href="./config-doc-cn.md">简体中文</a>  </center>

## Introduction

MetDetPy utilizes configuration files to read arguments. Generally, preset configuration files suffice for most scenarios. However, there might be instances where adjusting detection arguments can yield improved detection results. This document aims to elucidate the meanings of these arguments, enabling users to adjust them as per their requirements.

A configuration file is a `JSON` formatted text comprising four sections: [description](#description), [loader](#loader), [detector](#detector), and [Collector](#collector). For reference, predefined settings are stored in theFor reference, predefined settings are stored in theFor reference, predefined settings are stored in the [config](../config) directory. While going through this guide, it's advisable to use [the default configuration file](../config/m3det_normal.json) as a reference to better comprehend the configuration file format.

## Description

The `description` section provides the name of the configuration and a brief introduction. This section is primarily designed for frontend applications, as MetDetPy often operates as the backend. However, it's not critical for the detection process.

Here is an example of the `description` section:

```json
"description": {
    "preset_name": "configuration_name",
    "intro": "a_brief_introduction",
}
```

Furthermore, if you desire different content under varying language settings, you can format the `description` as follows:

```json
"description": {
    "preset_name_cn": "M3Det-通用",
    "preset_name_en": "M3Det-general",
    "intro_cn": "自适应检测器，适用于多数检测场景。",
    "intro_en": "Adaptive meteor detector. Suitable for most detection scenarios."
}
```

## Loader

The `loader` section primarily handles arguments related to video loading and frame preprocessing. Here's an example:

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

The loader `section` arguments are explained as follows:

|Argument|Type|Explanation|Recommendation|
|------|---|---|--------|
|name|str|Specifies the `loader` name. Options include `"VanillaVideoLoader"`,`"ThreadVideoLoader"`, and `ProcessVideoLoader`. `"ThreadVideoLoader"` is recommended due to its stability and speed.|`"ThreadVideoLoader"`|
|wrapper|str| Names the `wrapper` used for video loading. Currently, only `"OpenCVVideoWrapper"` based on OpenCV is supported. |`"OpenCVVideoWrapper"`|
|resize|int, array, str| Determines the resolution used during detection. Lower resolution speeds up detection, while higher resolution aids in detecting shorter meteors. Input an integer to specify the length of the long side, allowing the program to scale the video automatically(Recommended). Specifying both height and width simultaneously is also supported (input a string of two numbers separated by `:` or `x`, like `"960:540"` or `"960x540"`; a list format like `[960,540]` is also acceptable).  |`960`|
|exp_time|float, str(`"auto"`, `"slow"`, `"real-time"`)| Defines the exposure time of a single frame. For traditional detectors, setting `"auto"` is recommended as MetDetPy will estimate the real exposure time based on the video. If the exposure time in the metadata matches the real one, set `"real-time"`. To specify a value, input a float in seconds. |`"auto"` (LineDetector)/ `0.5` (MLDetector)|
|merge_func|str| Indicates how multiple frames are merged, choosing from `"max"`,`"m3func"`, or `"mix_max_median_stacker"`. Due to the visual characteristic of meteors, `"max"` is recommended.|`"max"`|
|grayscale|bool| Determines whether to load the video in grayscale. For LineDetector, `grayscale` must be set to `true`; for deep learning-based detector, it should be `false`. |`true`(LineDetector)/ `false`(MLDetector)|

⚠️ 
1.  `"name": "ProcessVideoLoader"` is still in the experimental stage and has not been fully tested. It is not compatible with MacOS and is not recommended for use in a production environment.
2. Q: Why is the exposure time needed? A: There are instances where the real exposure time and frames per second (FPS) do not align. For example, some cameras encode 4k60p videos with an exposure time of 1/20s. In these videos, frames change every 3 ticks instead of 1. Utilizing the real exposure time can enhance both accuracy and efficiency.

## Detector

The `detector` section manages the name of the detector, the length of the detection sliding-window, and other detector-related arguments (like binary threshold and line detection settings).

Here is an example for a line detector:

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

And Here is an example for deep learning-based detector:

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

These arguments are explained as follows:

|Argument|Type|Explanation|Recommendation|
|------|---|---|---|
|name|str|Specifies the detector's name. Options include `"ClassicDetector"`, `"M3Detector"`, and `"MLDetector"`. `"ClassicDetector"` and `"M3Detector"` are traditional detectors based on line detection, while `"MLDetector"` is a deep learning-based detector. Traditional detectors are typically faster and more sensitive, which may lead to more false-positive samples (this can be improved by configuring [recheck](#recheck_cfg)); on the other hand, deep learning-based detectors are more robust but have a higher computing cost. These two types of detectors also differ in their arguments. |`"M3Detector"`(traditional detector)/ `"MLDetector"` (deep learning detector)|
|windows_sec|int,float| Defines the time length of the sliding window. Meteors are detected within the sliding window during operation. Generally, it is recommended to set this around `1` second. |`1`|
|cfg|-|Describes the specific arguments of the detector. |(See below)|

When using traditional detectors, it's necessary to configure `"binary"`(binary threshold arguments), `"hough_line"`(line detection arguments) and `"dynamic"`(dynamic mechanism arguments) under the `cfg` of the `detector`. When using deep learning-based detectors, `"model"` needs to be configured under the `cfg` of the `detector`. The specific requirements of these arguments are as follows:


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
        <td>Indicates whether to apply the adaptive binary threshold algorithm. This algorithm estimates the optimal binary threshold based on the signal noise rate (SNR) of the sampling area. This is crucial for line detect-based detectors.</td> 
        <td>true</td> 
    </tr>
    <tr>
        <td>init_value</td>  
        <td>int</td> 
        <td>Specifies the initial binary threshold. If the adaptive binary threshold is not applied, this value will be used as the constant binary threshold.</td> 
        <td>5</td>  
    </tr>
    <tr>
        <td>sensitivity</td>  
        <td>str</td> 
        <td>Defines the sensitivity of the adaptive binary threshold algorithm. Different sensitivity functions will map the SNR value to the corresponding threshold. If the adaptive binary threshold is not applied, sensitivity will not be used. </td> 
        <td>normal</td>  
    </tr>
    <tr>
        <td>area</td>  
        <td>float</td> 
        <td>Indicates the sampling area size of the adaptive binary threshold algorithm. The range is [0,1], representing the area ratio of the whole image. Setting an appropriate value will expedite estimation while obtaining an approximate value. It is recommended to set it between 0.05 to 2. Illegal value (>1 or <0) setting will lead to using the whole area for estimation.  </td> 
        <td>0.1</td>  
    </tr>
    <tr>
        <td>interval</td>  
        <td>int</td> 
        <td>Sets the threshold estimation interval. Excessive threshold estimation can increase the computational load and reduce the performance of the program. Conversely, a too low computation frequency may not be able to promptly adjust according to the changes in the signal-to-noise ratio.</td> 
        <td>2</td>  
    </tr>
    <tr>
        <td rowspan="3"><p>hough_line</p>(Hough line detection related arguments. For specific meanings of the parameters, refer to the <a href="https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html">OpenCV official documentation</a>)</td>  
        <td>hough_threshold</td> 
        <td>int</td>  
        <td>The threshold for Hough line detection.</td>
        <td>10</td>  
    </tr>
    <tr>
        <td>min_len</td>  
        <td>int</td> 
        <td>Describes the minimum length of the line.</td> 
        <td>10</td>  
    </tr>
    <tr>
        <td>max_gap</td>  
        <td>int</td> 
        <td>Describes the maximum allowable gap between lines.</td> 
        <td>10</td>  
    </tr>
    <tr>
        <td rowspan="3"><p>dynamic</p>(dynamic mechanism related arguments)</td>  
        <td>dy_mask</td> 
        <td>bool</td>  
        <td>Indicates whether to enable the dynamic mask mechanism. The dynamic mask applies a mask to areas that continuously generate responses, reducing false positive samples in areas near bright stars.</td>
        <td>true</td>  
    </tr>
    <tr>
        <td>dy_gap</td>  
        <td>float</td> 
        <td>Dynamic gap parameter. This parameter reduces the "max_gap" in the Hough parameters when there are too many responses，reducing false positives. This value indicates that as the potential meteor area increases to a given value, the "max_gap" decays to 0. If set to 0.05, the "max_gap" decays to 0 when the potential meteor area is greater than 0.05%. </td> 
        <td>0.05</td>  
    </tr>
    <tr>
        <td>fill_thre</td>  
        <td>float</td> 
        <td>Describes the maximum allowable hollow ratio to form a line, used to reduce false alarms from discontinuous lines. Set to 0 to disable this.</td> 
        <td>0.6</td>  
    </tr>
    <tr>
        <td><p>model</p>(model related arguments)</td>  
        <td>-</td>  
        <td>-</td> 
        <td>See <a href=#model>model</a> section.</td> 
        <td>0.6</td>  
    </tr>
</table>

## Collector

The `collector` section manages the the filtering and collection conditions for meteors, and configuration about re-verification. Here's an example:

```json
"collector": {
    "meteor_cfg": {
        "min_len": 15,
        "max_interval": 5,
        "time_range": [
            0,
            8
        ],
        "speed_range": [
            0.9,
            18
        ],
        "drct_range": [
            0,
            0.5
        ],
        "det_thre": 0.5,
        "thre2": 2048
    },
    "recheck_cfg": {
        "switch": true,
        "model": {
            "name":"YOLOModel",
            "weight_path": "./weights/yolov5s.onnx",
            "dtype": "float32",
            "nms": true,
            "warmup": true,
            "pos_thre": 0.25,
            "nms_thre": 0.45
        },
        "save_path":""
    }
}
```

This mainly includes two settings: the filter configuration for meteors `"meteor_cfg"` and the re-verification configuration `"recheck_cfg"`.

### Meteor_cfg

The `meteor_cfg` allows setting filters for the speed, duration, allowable interval, linearity, and score threshold of the meteors. The explanations of each parameter are as follows:

|Argument|Type|Explanation|Recommendation|
|------|---|---|---|
|min_len|int|The minimum length (px) required to start recording a meteor.|10|
|max_interval|int|The longest time interval between meteors. (if there is no other responses after this time interval, this meteor (group) is considered to be ended). Unit: s.|5|
|time_range|array|Describes the duration range of the meteor. Responses that exceed or do not reach the threshold will be excluded. Unit: s.|[0,8]|
|speed_range|array|Describes the allowable speed range of the meteor. Responses that exceed or do not reach the threshold will be excluded. Unit: $f^{-1}$.|[0.9, 18]|
|drct_range|array|Describes the linearity range of the meteor. The closer to 0, the closer the meteor is to a perfect straight line.|[0,0.5]|
|det_thre|float|Describes the threshold for a positive sample meteor. Responses that exceed this score are considered positive sample meteors. Range: [0,1].|0.5|
|thre2|int|Describes the maximum allowable square distance between several responses. If there are multiple responses for one trajectory in the detection results, consider increasing this value.|2048|

⚠️
1. The current filtering adopts a tolerant design: when values of response exceed the above ranges, the score will not immediately drop to zero, but with a gradually decay.
2. The design of the above filtering thresholds is still based on the runtime resolution. When using a resolution that differs from the default one, these parameters may produce inaccurate results.
3. "Responses" refer to all meteor candidates that are captured by detectors. Only responses that have proper motion properties and visual appearances are considered as meteors (or other categories).


### Recheck_cfg

`Recheck_cfg` is a mechanism introduced in `v2.0.0` that
executes additional re-verification for proposed target objects (such as meteors, red sprites, etc.) from the main detector. The explanations of itsparameters are as follows:

|Argument|Type|Explanation|Recommendation|
|------|---|---|---|
|switch|str|Whether re-verification is enabled.|true|
|model|-|Refer to the [Model](#Model) section.|-|
|save_path|str|The path to save the re-verified images. If left blank, no image will be saved.|`""`|


## Model

Models can be configured in the `cfg` of the `detector` and the `recheck_cfg` section of the `collector`. Here's an example of configuring a `model`:

```json
"model": {
    "name":"YOLOModel",
    "weight_path": "./weights/yolov5s.onnx",
    "dtype": "float32",
    "nms": true,
    "warmup": true,
    "pos_thre": 0.25,
    "nms_thre": 0.45
}
```

|Argument|Type|Explanation|Recommendation|
|------|---|---|---|
|name|str|The type of deep learning model, which will determine how the program processes input and output. Currently, only the YOLO format model `"YOLOModel"` has been implemented.|`"YOLOModel"`|
|weight_path|str|The path to the model weights. It can be a path relative to MetDetPy or an absolute path. A trained YOLOv5s is provided in the project. The labels of the network output should refer to the [class_name file](../config/class_name.txt). Currently, the `.onnx` network weight format is supported.|`"./weights/yolov5s.onnx"`|
|dtype|str|Describes the input data type (dtype) of the network. When using a quantized model, configure the specific dtype here to make sure the network work properly. Currently, full precision (`"float32"`) and half precision (`"float16"`) dtype are supported.|`"float32"`|
|nms|bool|Desribes whether Non-Maximum Suppression (NMS) deduplication needs to be executed. If the network already includes NMS, select `false` to accelerate.|`true`|
|warmup|bool|Desribes whether to warmup before execution. Set to `true` to accelerate the execution.|`true`|
|pos_thre|float|Desribes the positive sample threshold. Scores exceeding the threshold will be considered as positive samples. Range: [0,1].|0.25|
|nms_thre|float|The threshold used for NMS deduplication.|0.45|