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
python evaluate.py --report REPORT [--cfg CFG] [--load LOAD] [--save-path SAVE_PATH] [--metric] [--debug]
```

### Arguments

* `--report`: A JSON file in `MDRF` format, which needs to contain the necessary information related to the video (video file and mask file paths, start and end times) to initiate. Its format should meet the requirements specified in [Meteor Detection Recording Format (MDRF)](#meteor-detection-recording-format-mdrf).

* `--cfg`: Configuration file. By default, it uses the default configuration, which is [m3det_normal.json](../config/m3det_normal.json).

* `--load`: If this is enabled with a path to another `JSON`, `evaluate.py` will directly load its results for comparison as the current detection result instead of running detection through the video.

* `--save`: The path and filename where the detection results will be saved.

* `--metric`: Depending on the category of the provided JSON file, it performs regression testing (comparing with other prediction results) or calculates detection precision and recall (comparing with ground truth). To apply this option, the `json` file needs to contain `results` information.

* `--debug`: When starting `evaluate.py` with this option, detailed debug information will be provided.

`--manifest MANIFEST` accepts a JSON object containing a `cases` list instead of an MDRF report. Exactly one of `--report` and `--manifest` is required. Use repeatable `--case ID` to select manifest cases and `--passes N` for repeated runs (default: 1).

### Example
(To be updated)

---

## make_package

Use [make_package.py](../make_package.py) to freeze the MetDetPy programs with
either Nuitka or PyInstaller. Nuitka is the default backend.

```sh
python make_package.py [--backend {nuitka,pyinstaller}]
     [--apply-upx] [--apply-zip] [--onefile]
     [--mingw64] [--macos-sign-identity IDENTITY]
     [--windowed] [--icon ICON_PATH]
```

* `--backend`: select `nuitka` or `pyinstaller`. Defaults to `nuitka`.

* `--apply-upx`: apply UPX to squeeze the size of the executable program.

* `--apply-zip`: generate a ZIP package after packaging.

* `--onefile`: generate one executable per program instead of a directory bundle.

* `--mingw64`: use MinGW64 on Windows with Python 3.12 or earlier. Python 3.13+ automatically uses MSVC; install Visual Studio Build Tools with the Desktop development with C++ workload.

* `--macos-sign-identity`: macOS signing identity for the Nuitka backend.

* `--windowed`: use windowed mode with the PyInstaller backend.

* `--icon`: executable icon used by the PyInstaller backend.

### Common Notes

Executables and the optional ZIP package are generated in the [dist](../dist/) directory.

**Notice:**

1. Use a Nuitka release supporting your Python version and an available C/C++ compiler. Python 3.13 support started in Nuitka 2.5. For PyInstaller, install `pyinstaller>=6.0`.
2. Due to the nature of Python packaging, these tools cannot generate cross-platform executables; build the executable on the target platform.
3. PyInstaller excludes optional plotting, training, interactive development, and Qt dependencies from the three runtime tools. Development photo comparison plots are not part of the release bundle. A dedicated virtual environment is still recommended; required native libraries such as `pyexiv2` remain included.

Use a separate environment populated from the runtime dependency list, for example on Windows:

```powershell
python -m venv .venv-package
.venv-package\Scripts\python.exe -m pip install -r requirements.txt pyinstaller
.venv-package\Scripts\python.exe make_package.py --backend pyinstaller
```

This list controls direct build-environment dependencies; pip installs their required transitive dependencies and PyInstaller analyzes them. It does not truncate the final module graph. For a Windows DML release, use `onnxruntime-directml` instead of generic `onnxruntime` in that environment.
