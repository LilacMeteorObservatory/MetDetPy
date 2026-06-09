# Nuitka packaging backend.

import os
import platform as pf
import shutil
import sys
from typing import Union

from MetLib.utils import PROJECT_NAME, VERSION, PLATFORM_MAPPING
from .common import run_cmd, post_process

join_path = os.path.join

SCRIPTS = ["MetDetPy.py", "ClipToolkit.py", "MetDetPhoto.py"]

EXCLUDE_PKGS = ["torch", "scipy", "tensorflow", "Ipython", "Keras", "PIL"]


def nuitka_compile(header: list[str], options: dict[str, Union[bool, str]],
                   nuitka_pkgs: list[str], target: str):
    options_list = [
        key if value is True else f'{key}={value}'
        for key, value in options.items() if value
    ]
    merged = header + options_list + nuitka_pkgs + [target]

    ret_code, time_cost = run_cmd(merged)
    print(
        f"Compiled {target} finished with return code = {ret_code}. "
        f"Time cost = {time_cost:.2f}s.")

    if ret_code != 0:
        print(
            f"Fatal compile error occured when compiling {target}. "
            f"Compile terminated.")
        exit(-1)


def build(args):
    platform = PLATFORM_MAPPING[sys.platform]
    exec_suffix = ""
    if platform == "win":
        exec_suffix = ".exe"
    if platform == "macos":
        mac_main_ver = int(pf.mac_ver()[0].split(".")[0])
        if mac_main_ver >= 13:
            exec_suffix = ".bin"
            platform += "13+"

    work_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    compile_path = join_path(work_path, "dist")

    print("Use nuitka as package tool.")

    compile_tool = [sys.executable, "-m", "nuitka"]

    nuitka_base: dict[str, Union[bool, str]] = {
        "--no-pyi-file": True,
        "--remove-output": True,
        "--lto": "yes"
    }

    nuitka_pkgs = [f"--nofollow-import-to={x}" for x in EXCLUDE_PKGS]
    nuitka_pkgs.append("--include-package-data=pyexiv2")

    if platform == "win" and args.mingw64:
        print("Apply mingw64 as compiler.")
        nuitka_base["--mingw64"] = True

    if platform.startswith("macos"):
        nuitka_base["--macos-app-version"] = VERSION
        nuitka_base[
            "--macos-signed-app-name"] = "org.lilacMeteorobservatory.metdetpy"
        if args.macos_sign_identity:
            nuitka_base["--macos-sign-identity"] = args.macos_sign_identity

    if args.apply_upx:
        upx_cmd = shutil.which("upx")
        if upx_cmd is not None:
            nuitka_base["--plugin-enable"] = "upx"
            nuitka_base["--upx-binary"] = upx_cmd

    onefile_mode = args.onefile
    if onefile_mode:
        print("WARNING: onefile mode may have issues with static file paths.")
        print("Consider using directory mode (default) instead.")

    for script in SCRIPTS:
        cfg: dict[str, Union[bool, str]] = {
            "--standalone": True,
            "--output-dir": compile_path,
        }
        if onefile_mode:
            cfg["--onefile"] = True
        cfg.update(nuitka_base)

        nuitka_compile(compile_tool, cfg, nuitka_pkgs,
                       target=join_path(work_path, script))

    # Post-compile: merge dist folders or cleanup
    if onefile_mode:
        print("Cleaning up...", end="", flush=True)
        for script in SCRIPTS:
            dist_dir = join_path(compile_path,
                                script.replace(".py", ".dist"))
            try:
                shutil.rmtree(dist_dir)
            except FileNotFoundError:
                pass
        print("Done.")
    else:
        print("Merging...", end="", flush=True)
        for script in SCRIPTS[1:]:
            base = script.replace(".py", "")
            shutil.move(
                join_path(compile_path, f"{base}.dist",
                          f"{base}{exec_suffix}"),
                join_path(compile_path, "MetDetPy.dist"))
            shutil.rmtree(join_path(compile_path, f"{base}.dist"))
        shutil.move(join_path(compile_path, "MetDetPy.dist"),
                    join_path(compile_path, "MetDetPy"))
        print("Done.")

    post_process(compile_path, onefile_mode, args.apply_zip)
