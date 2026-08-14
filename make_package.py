# 简易的打包工具
# 用于将本项目封装为一个（数个）可执行文件。

import argparse
import sys
import time

argparser = argparse.ArgumentParser(
    description="Package MetDetPy into standalone executables.")

argparser.add_argument(
    "--backend",
    "-B",
    help="Packaging backend.",
    choices=["nuitka", "pyinstaller"],
    default="nuitka",
    type=str,
)

# Shared options
argparser.add_argument(
    "--apply-zip",
    action="store_true",
    help="Generate .zip files after packaging.",
)
argparser.add_argument(
    "--onefile",
    action="store_true",
    help="Generate a single executable file (onefile mode).",
)
argparser.add_argument(
    "--apply-upx",
    action="store_true",
    help="Apply UPX to squeeze executable size.",
)

# Nuitka-specific options
nuitka_group = argparser.add_argument_group("nuitka options")
nuitka_group.add_argument(
    "--mingw64",
    action="store_true",
    help="Use mingw64 as compiler (Windows + nuitka only).",
    default=False,
)
nuitka_group.add_argument(
    "--macos-sign-identity",
    type=str,
    help="macOS code signing identity (nuitka only).",
)

# PyInstaller-specific options
pyinstaller_group = argparser.add_argument_group("pyinstaller options")
pyinstaller_group.add_argument(
    "--windowed",
    action="store_true",
    help="Use windowed mode / no console (pyinstaller only).",
)
pyinstaller_group.add_argument(
    "--icon",
    type=str,
    help="Icon file path for the executable (pyinstaller only).",
)

args = argparser.parse_args()

if args.apply_upx and sys.platform == "darwin":
    print("WARNING: UPX breaks code signatures on macOS. --apply-upx ignored.")
    args.apply_upx = False

t0 = time.time()

if args.backend == "nuitka":
    from MetLib.packaging.nuitka import build
elif args.backend == "pyinstaller":
    from MetLib.packaging.pyinstaller import build

build(args)

print(f"Package script finished. Total time cost {(time.time()-t0):.2f}s.")
