# 简易的打包工具
# 用于将本项目封装为一个（数个）可执行文件。
# 使用 pyinstaller 打包
# 使用 MERGE 机制共享依赖，避免重复打包。

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from MetLib.utils import PROJECT_NAME, VERSION, PLATFORM_MAPPING

join_path = os.path.join

SCRIPTS = [
    ("MetDetPy.py", "MetDetPy"),
    ("ClipToolkit.py", "ClipToolkit"),
    ("MetDetPhoto.py", "MetDetPhoto"),
]

HIDDEN_IMPORTS = [
    "cv2",
    "numpy",
    "PIL",
    "PIL.Image",
    "pyexiv2",
    "uuid",
    "pyexpat",
    "xml.etree.ElementTree",
]

EXCLUDES = ["torch", "scipy", "tensorflow", "Ipython", "Keras", "pkg_resources"]


def run_cmd(command: list[str]):
    print("Running:", command)
    t_start = time.time()
    ret = subprocess.run(command)
    t_end = time.time()
    return ret.returncode, t_end - t_start


def file_to_zip(path_original: str, z: zipfile.ZipFile):
    f_list = list(Path(path_original).glob("**/*"))
    for f in f_list:
        z.write(f, str(f)[len(path_original):])


def copy_tree(tree_path: str, tgt_path: str):
    print(f"  {tree_path}...", end="", flush=True)
    tgt_dir = f"{tgt_path}/{tree_path}"
    if os.path.exists(tgt_dir):
        print("exists, skipped.")
        return
    shutil.copytree(f"./{tree_path}", tgt_dir)
    print("ok.")


def create_merged_spec(work_path: str, onefile: bool, console: bool,
                       icon_path: str = None) -> str:
    """Generate a single .spec file that uses MERGE to share dependencies."""
    hidden_imports_str = str(HIDDEN_IMPORTS)
    excludes_str = str(EXCLUDES)
    icon_line = f"    icon='{icon_path}'," if icon_path else ""

    # Build Analysis blocks
    analysis_blocks = []
    for script_name, script_base in SCRIPTS:
        analysis_blocks.append(f"""
{script_base}_a = Analysis(
    ['{join_path(work_path, script_name)}'],
    pathex=['{work_path}'],
    binaries=[],
    datas=[],
    hiddenimports={hidden_imports_str},
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes={excludes_str},
    noarchive=False,
)""")

    # MERGE deduplicates shared binaries across all Analysis objects
    merge_args = ", ".join(
        f"({base}_a, '{base}', '{base}')" for _, base in SCRIPTS
    )

    # Build PYZ + EXE blocks
    exe_blocks = []
    for _, script_base in SCRIPTS:
        exe_blocks.append(f"""
{script_base}_pyz = PYZ({script_base}_a.pure)

{script_base}_exe = EXE(
    {script_base}_pyz,
    {script_base}_a.scripts,
    [],
    name='{script_base}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console={console},
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
{icon_line}
)""")

    if onefile:
        # Onefile: each EXE bundles its own binaries/datas (no COLLECT)
        exe_blocks_onefile = []
        for _, script_base in SCRIPTS:
            exe_blocks_onefile.append(f"""
{script_base}_pyz = PYZ({script_base}_a.pure)

{script_base}_exe = EXE(
    {script_base}_pyz,
    {script_base}_a.scripts,
    {script_base}_a.binaries,
    {script_base}_a.datas,
    [],
    name='{script_base}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console={console},
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
{icon_line}
)""")

        spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
{"".join(analysis_blocks)}

MERGE({merge_args})
{"".join(exe_blocks_onefile)}
"""
    else:
        # Directory mode: shared COLLECT with all EXEs and deduplicated deps
        collect_entries = []
        for _, script_base in SCRIPTS:
            collect_entries.append(f"    {script_base}_exe,")
            collect_entries.append(f"    {script_base}_a.binaries,")
            collect_entries.append(f"    {script_base}_a.datas,")

        spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
{"".join(analysis_blocks)}

MERGE({merge_args})
{"".join(exe_blocks)}

coll = COLLECT(
{chr(10).join(collect_entries)}
    strip=False,
    upx=False,
    upx_exclude=[],
    name='{PROJECT_NAME}',
)
"""

    spec_file = join_path(work_path, f"{PROJECT_NAME}.spec")
    with open(spec_file, "w", encoding="utf-8") as f:
        f.write(spec_content)
    return spec_file


argparser = argparse.ArgumentParser()

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
    "--windowed",
    action="store_true",
    help="Use windowed mode (no console) for GUI applications.",
)
argparser.add_argument(
    "--icon",
    type=str,
    help="Icon file path for the executable.",
)

args = argparser.parse_args()
release_version = VERSION
apply_zip = args.apply_zip
onefile_mode = args.onefile
console = not args.windowed
icon_path = args.icon

platform = PLATFORM_MAPPING[sys.platform]
exec_suffix = ".exe" if platform == "win" else ""

work_path = os.path.dirname(os.path.abspath(__file__))
if not work_path:
    work_path = os.getcwd()
compile_path = join_path(work_path, "dist")

t0 = time.time()

print("Using pyinstaller as package tool.")
os.makedirs(compile_path, exist_ok=True)

# Generate single merged spec and compile once
spec_file = create_merged_spec(work_path, onefile_mode, console, icon_path)
cmd = [sys.executable, "-m", "PyInstaller", spec_file]
ret_code, time_cost = run_cmd(cmd)
print(f"Build finished with return code = {ret_code}. Time cost = {time_cost:.2f}s.")
if ret_code != 0:
    print("Fatal compile error. Terminated.")
    exit(-1)

# Cleanup build artifacts
print("Cleaning up build artifacts...", end="", flush=True)
if os.path.exists(spec_file):
    os.remove(spec_file)
build_root = join_path(work_path, "build")
if os.path.exists(build_root):
    shutil.rmtree(build_root)
print("Done.")

# Copy static folders
print("Copying static folders:")
src_list = ["config", "weights", "global"]
tgt_base = join_path(compile_path, PROJECT_NAME) if not onefile_mode else compile_path
for src_folder in src_list:
    if os.path.exists(src_folder):
        copy_tree(src_folder, tgt_base)

# Copy uuid module (needed at runtime)
import uuid
shutil.copy(uuid.__file__, tgt_base)

# Copy pyexiv2 package
try:
    import pyexiv2
    pyexiv_path, _ = os.path.split(pyexiv2.__file__)
    pyexiv_tgt = join_path(tgt_base, "pyexiv2")
    if os.path.exists(pyexiv_tgt):
        shutil.rmtree(pyexiv_tgt)
    shutil.copytree(pyexiv_path, pyexiv_tgt)
except Exception as e:
    print(f"  pyexiv2 copy skipped: {e}")

# Zip output
if apply_zip:
    zip_fname = join_path(compile_path, f"MetDetPy_{platform}_{release_version}.zip")
    print(f"Zipping files to {zip_fname} ...", end="", flush=True)
    with zipfile.ZipFile(zip_fname, mode='w') as zipfile_op:
        file_to_zip(join_path(compile_path, PROJECT_NAME), zipfile_op)
    print("Done.")

print(f"Package script finished. Total time cost {(time.time()-t0):.2f}s.")
