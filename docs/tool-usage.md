# Tools Usage

<center>Language: English | <a href="./tool-usage-cn.md">简体中文</a>  </center>

Several tools are provided with MetDetPy to support related functions.

## Menu

### Tool User Guides
* [Detection Tools - MetDetPy and MetDetPhoto](#detection-tools)
* [ClipToolkit - (Batch) image stacking and video clipping](#cliptoolkit)

### Other Tools
* [Evaluate - Performance evaluation and regression testing](#evaluate)
* [make_package - Packaging script to executable files](#make-package)

### Data Format
* [Meteor Detection Recording Format (MDRF)](#meteor-detection-recording-format-mdrf)

## Tool User Guides

### Detection Tools

MetDetPy provides two detection tools: `MetDetPy` for video meteor detection and `MetDetPhoto` for image meteor detection. Each tool has its own characteristics and is suitable for different use cases.

For detailed usage information about these detection tools, please refer to the [Detection Tools User Guide](./tool-usage/Detector-usage.md).

---

### ClipToolkit

`ClipToolkit` can be used to create multiple video segments from a single video or a stack of images from these video segments at once. For detailed usage information, please see the [ClipToolkit User Guide](./tool-usage/ClipToolkit-usage.md).

---

## Evaluate

Evaluate is an integrated performance evaluation and regression testing tool. It can be used to generate result reports, evaluate the utilization of device resources, and compare differences between results.

To evaluate how MetDetPy performs on your video, you can simply run `evaluate.py` :

```sh
python evaluate.py json [--cfg CFG] [--load LOAD] [--save SAVE] [--metrics] [--debug]
```

### Arguments

* `json`: A JSON file in `MDRF` format, which needs to contain the necessary information related to the video (video file and mask file paths, start and end times) to initiate. Its format should meet the requirements specified in [Meteor Detection Recording Format (MDRF)](#meteor-detection-recording-format-mdrf).

* `--cfg`: Configuration file. By default, it uses the default configuration, which is [m3det_normal.json](../config/m3det_normal.json).

* `--load`: If this is enabled with a path to another `JSON`, `evaluate.py` will directly load its results for comparison as the current detection result instead of running detection through the video.

* `--save`: The path and filename where the detection results will be saved.

* `--metrics`: Depending on the category of the provided JSON file, it performs regression testing (comparing with other prediction results) or calculates detection precision and recall (comparing with ground truth). To apply this option, the `json` file needs to contain `results` information.

* `--debug`: When starting `evaluate.py` with this option, detailed debug information will be provided.

### Example
(To be updated)

---

## make_package

We provide two packaging scripts to freeze MetDetPy programs into stand-alone executables:

1. **[make_package.py](../make_package.py)** - Uses `nuitka` for compilation
2. **[make_package_pyinstaller.py](../make_package_pyinstaller.py)** - Uses `pyinstaller` for packaging

Choose the one that best fits your needs.

### Using Nuitka (make_package.py)

When using `nuitka`, ensure at least one C/C++ compiler is available on your computer.

```sh
python make_package.py [--mingw64] [--apply-upx] [--apply-zip]
     [--onefile]
```

* `--mingw64`: use the mingw64 compiler. Only works on Windows.

* `--apply-upx`: apply UPX to squeeze the size of the executable program.

* `--apply-zip`: generate zip package when compilation is finished.

* `--onefile`: generate a single executable file (onefile mode). When using this mode, you need to ensure that the static resource folders (`config/`, `weights/`, `resource/`, `global/`) are placed next to the executable file, or use the `--resource-dir` / `-R` option to specify their location at runtime.

### Using PyInstaller (make_package_pyinstaller.py)

```sh
python make_package_pyinstaller.py [--apply-zip] [--onefile]
     [--windowed] [--icon ICON_PATH]
```

* `--apply-zip`: generate zip package when packaging is finished.

* `--onefile`: generate a single executable file (onefile mode).

* `--windowed`: use windowed mode (no console) for GUI applications.

* `--icon`: path to an icon file for the executable.

### Common Notes

The target executable file and its zip package version (if applied) will be generated in the [dist](./dist/) directory.

**Notice:**

1. It is suggested to use `Python>=3.9` and `nuitka>=2.0.0` when using `nuitka`. For `pyinstaller`, ensure `pyinstaller>=6.0`.
2. Due to the nature of Python packaging, these tools cannot generate cross-platform executables; build the executable on the target platform.
3. If `matplotlib` or `scipy` exists in the environment, they may be included in the packaged output. To reduce package size, prepare a clean environment or avoid installing heavy optional dependencies.
