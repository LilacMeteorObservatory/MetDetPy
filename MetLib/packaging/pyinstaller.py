# PyInstaller packaging backend.

import os
import shutil
import sys
from typing import Optional

from MetLib.utils import PROJECT_NAME, VERSION, PLATFORM_MAPPING
from .common import run_cmd, post_process

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


def create_merged_spec(work_path: str, onefile: bool, console: bool,
                       icon_path: Optional[str] = None,
                       upx: bool = False) -> str:
    hidden_imports_str = str(HIDDEN_IMPORTS)
    excludes_str = str(EXCLUDES)
    icon_line = f"    icon={icon_path!r}," if icon_path else ""

    # collect_all preamble for native packages PyInstaller can't auto-trace
    collect_preamble = """
from PyInstaller.utils.hooks import collect_all

_pyexiv2_datas, _pyexiv2_binaries, _pyexiv2_hiddenimports = collect_all('pyexiv2')
"""

    analysis_blocks = []
    for script_name, script_base in SCRIPTS:
        script_path = join_path(work_path, script_name)
        analysis_blocks.append(f"""
{script_base}_a = Analysis(
    [{script_path!r}],
    pathex=[{work_path!r}],
    binaries=_pyexiv2_binaries,
    datas=_pyexiv2_datas,
    hiddenimports={hidden_imports_str} + _pyexiv2_hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes={excludes_str},
    noarchive=False,
)""")

    merge_args = ", ".join(
        f"({base}_a, '{base}', '{base}')" for _, base in SCRIPTS
    )

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
    upx={upx},
    console={console},
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
{icon_line}
)""")

    if onefile:
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
    upx={upx},
    console={console},
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
{icon_line}
)""")

        spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
{collect_preamble}
{"".join(analysis_blocks)}

{"".join(exe_blocks_onefile)}
"""
    else:
        collect_entries = []
        for _, script_base in SCRIPTS:
            collect_entries.append(f"    {script_base}_exe,")
            collect_entries.append(f"    {script_base}_a.binaries,")
            collect_entries.append(f"    {script_base}_a.datas,")

        spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
{collect_preamble}
{"".join(analysis_blocks)}

MERGE({merge_args})
{"".join(exe_blocks)}

coll = COLLECT(
{chr(10).join(collect_entries)}
    strip=False,
    upx={upx},
    upx_exclude=[],
    name='{PROJECT_NAME}',
)
"""

    spec_file = join_path(work_path, f"{PROJECT_NAME}.spec")
    with open(spec_file, "w", encoding="utf-8") as f:
        f.write(spec_content)
    return spec_file


def build(args):
    platform = PLATFORM_MAPPING[sys.platform]

    work_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    compile_path = join_path(work_path, "dist")

    onefile_mode = args.onefile
    console = not args.windowed
    icon_path = args.icon

    print("Using pyinstaller as package tool.")
    os.makedirs(compile_path, exist_ok=True)

    upx = args.apply_upx
    spec_file = create_merged_spec(work_path, onefile_mode, console, icon_path, upx)
    build_root = join_path(work_path, "build")
    cmd = [sys.executable, "-m", "PyInstaller",
           "--distpath", compile_path,
           "--workpath", build_root,
           spec_file]
    ret_code, time_cost = run_cmd(cmd)
    print(f"Build finished with return code = {ret_code}. "
          f"Time cost = {time_cost:.2f}s.")
    if ret_code != 0:
        print("Fatal compile error. Terminated.")
        exit(-1)

    # Cleanup build artifacts
    print("Cleaning up build artifacts...", end="", flush=True)
    if os.path.exists(spec_file):
        os.remove(spec_file)
    if os.path.exists(build_root):
        shutil.rmtree(build_root)
    print("Done.")

    post_process(compile_path,
                 onefile_mode,
                 args.apply_zip,
                 source_root=work_path)
