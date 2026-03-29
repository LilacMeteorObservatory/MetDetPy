# 简易的打包工具
# 用于将本项目封装为一个（数个）可执行文件。
# 使用 pyinstaller 打包

import argparse
import os
import platform as pf
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Union

from MetLib.utils import PROJECT_NAME, VERSION, PLATFORM_MAPPING

join_path = os.path.join


def run_cmd(command: list[str]):
    print("Running:", command)
    t_start = time.time()
    ret = subprocess.run(command)
    t_end = time.time()
    return ret.returncode, t_end - t_start


def pyinstaller_compile(spec_file: str):
    cmd = [sys.executable, "-m", "PyInstaller", spec_file]
    ret_code, time_cost = run_cmd(cmd)
    print(f"Compiled {spec_file} finished with return code = {ret_code}. Time cost = {time_cost:.2f}s.")
    if ret_code != 0:
        print(f"Fatal compile error occured when compiling {spec_file}. Compile terminated.")
        exit(-1)


def file_to_zip(path_original: str, z: zipfile.ZipFile):
    f_list = list(Path(path_original).glob("**/*"))
    for f in f_list:
        z.write(f, str(f)[len(path_original):])


def copy_tree(tree_path: str, tgt_path: str):
    print(f"Copy {tree_path} folder...", end="", flush=True)
    tgt_dir = f"{tgt_path}/{tree_path}"
    if os.path.exists(tgt_dir):
        print("Already exists, skipped.")
        return
    shutil.copytree(f"./{tree_path}", tgt_dir)
    print("Done.")


def get_hidden_imports() -> list[str]:
    hidden_imports = [
        "cv2",
        "numpy",
        "PIL",
        "PIL.Image",
        "PIL.ImageTk",
        "pyexiv2",
        "uuid",
        "pyexpat",
        "xml.etree.ElementTree",
        "pkg_resources",
        "pkg_resources.extern",
    ]
    return hidden_imports


def create_spec_file(script_name: str, script_base: str, hidden_imports: list[str],
                    onefile: bool, console: bool, icon_path: str = None):
    import sys
    spec_file = join_path(work_path, f"{script_base}.spec")
    
    hidden_imports_str = str(hidden_imports).replace("'", '"')
    
    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

import sys
import os

block_cipher = None

a = Analysis(
    ['{join_path(work_path, script_name)}'],
    pathex=['{work_path}'],
    binaries=[],
    datas=[
        ('config', 'config'),
        ('weights', 'weights'),
        ('resource', 'resource'),
        ('global', 'global'),
    ],
    hiddenimports={hidden_imports_str},
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=['torch', 'scipy', 'tensorflow', 'Ipython', 'Keras', 'pkg_resources'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
"""
    
    if onefile:
        icon_line = f"    icon='{icon_path}'," if icon_path else ""
        spec_content += f"""
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
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
)
"""
    else:
        icon_line = f"    icon='{icon_path}'," if icon_path else ""
        spec_content += f"""
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='{script_base}',
)
"""
    
    with open(spec_file, "w", encoding="utf-8") as f:
        f.write(spec_content)
    
    return spec_file


argparser = argparse.ArgumentParser()

argparser.add_argument(
    "--tool",
    "-T",
    help="compiler. Only pyinstaller is available.",
    choices=['pyinstaller'],
    default='pyinstaller',
    type=str)
argparser.add_argument(
    "--apply-upx",
    action="store_true",
    help="Apply UPX to squeeze the size of executable program.",
)
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
windowed_mode = args.windowed
icon_path = args.icon

platform = PLATFORM_MAPPING[sys.platform]
exec_suffix = ".exe" if platform == "win" else ""

work_path = os.path.dirname(os.path.abspath(__file__))
if not work_path:
    work_path = os.getcwd()
compile_path = join_path(work_path, "dist")

t0 = time.time()

print("Using pyinstaller as package tool.")

hidden_imports = get_hidden_imports()

os.makedirs(compile_path, exist_ok=True)

scripts = [
    ("MetDetPy.py", "MetDetPy"),
    ("ClipToolkit.py", "ClipToolkit"),
    ("MetDetPhoto.py", "MetDetPhoto"),
]

console = not windowed_mode

spec_files = []
for script_name, script_base in scripts:
    spec_file = create_spec_file(script_name, script_base, hidden_imports, onefile_mode, console, icon_path)
    spec_files.append((spec_file, script_base))

for spec_file, script_base in spec_files:
    pyinstaller_compile(spec_file)

if onefile_mode:
    print("Cleaning up...", end="", flush=True)
    for script_base in ["MetDetPy", "ClipToolkit", "MetDetPhoto"]:
        spec_path = join_path(work_path, f"{script_base}.spec")
        if os.path.exists(spec_path):
            os.remove(spec_path)
        build_path = join_path(work_path, "build", script_base)
        try:
            if os.path.exists(build_path):
                shutil.rmtree(build_path)
        except FileNotFoundError:
            pass
    print("Done.")
else:
    print("Merging...", end="", flush=True)
    for script_base in ["ClipToolkit", "MetDetPhoto"]:
        src_path = join_path(compile_path, f"{script_base}.dist", f"{script_base}{exec_suffix}")
        tgt_path = join_path(compile_path, "MetDetPy.dist")
        if os.path.exists(src_path):
            shutil.move(src_path, tgt_path)
        dist_path = join_path(compile_path, f"{script_base}.dist")
        if os.path.exists(dist_path):
            shutil.rmtree(dist_path)
        spec_file = join_path(work_path, f"{script_base}.spec")
        if os.path.exists(spec_file):
            os.remove(spec_file)
    
    main_spec = join_path(work_path, "MetDetPy.spec")
    if os.path.exists(main_spec):
        os.remove(main_spec)
    
    build_path = join_path(work_path, "build", "MetDetPy")
    try:
        if os.path.exists(build_path):
            shutil.rmtree(build_path)
    except FileNotFoundError:
        pass
    
    print("Done.")
    print("Renaming executable folder...", end="", flush=True)
    shutil.move(join_path(compile_path, "MetDetPy.dist"), join_path(compile_path, "MetDetPy"))
    print("Done.")

print("Copying static folders...", end="", flush=True)
src_list = ["config", "weights", "resource", "global"]
tgt_base = "./dist/MetDetPy" if not onefile_mode else "./dist"
for src_folder in src_list:
    if os.path.exists(src_folder):
        copy_tree(src_folder, tgt_base)
print("Done.")

import uuid
uuid_tgt = "./dist/MetDetPy" if not onefile_mode else "./dist"
shutil.copy(uuid.__file__, uuid_tgt)

try:
    import pyexiv2
    pyexiv_path, _ = os.path.split(pyexiv2.__file__)
    pyexiv_tgt = "./dist/MetDetPy/pyexiv2" if not onefile_mode else "./dist/pyexiv2"
    if os.path.exists(pyexiv_tgt):
        shutil.rmtree(pyexiv_tgt)
    shutil.copytree(pyexiv_path, pyexiv_tgt)
except Exception as e:
    print(f"pyexiv2 copy skipped: {e}")

if apply_zip:
    zip_fname = join_path(compile_path, f"MetDetPy_{platform}_{release_version}.zip")
    print(f"Zipping files to {zip_fname} ...", end="", flush=True)
    with zipfile.ZipFile(zip_fname, mode='w') as zipfile_op:
        file_to_zip(join_path(compile_path, "MetDetPy"), zipfile_op)
    print("Done.")

print(f"Package script finished. Total time cost {(time.time()-t0):.2f}s.")
