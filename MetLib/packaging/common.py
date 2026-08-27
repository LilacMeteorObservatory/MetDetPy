# Shared utilities for packaging backends.

import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Optional

from MetLib.utils import PROJECT_NAME, VERSION, PLATFORM_MAPPING


def run_cmd(command: list[str]):
    print("Running:", command)
    t_start = time.time()
    ret = subprocess.run(command)
    t_end = time.time()
    return ret.returncode, t_end - t_start


def file_to_zip(path_original: str,
                z: zipfile.ZipFile,
                archive_root: Optional[str] = None):
    """Add a file or directory tree to ``z`` using relative archive names."""
    source = Path(path_original).resolve()
    root = Path(archive_root).resolve() if archive_root else source.parent
    zip_path = Path(z.filename).resolve() if z.filename else None
    files = [source] if source.is_file() else source.rglob("*")
    for file_path in files:
        if not file_path.is_file() or (zip_path is not None
                                       and file_path.resolve() == zip_path):
            continue
        z.write(file_path, file_path.resolve().relative_to(root).as_posix())


def copy_tree(tree_path: str, tgt_path: str, source_root: str = "."):
    print(f"  {tree_path}...", end="", flush=True)
    tgt_dir = os.path.join(tgt_path, tree_path)
    if os.path.exists(tgt_dir):
        print("exists, skipped.")
        return
    shutil.copytree(os.path.join(source_root, tree_path), tgt_dir)
    print("ok.")


def post_process(compile_path: str,
                 onefile_mode: bool,
                 apply_zip: bool,
                 source_root: Optional[str] = None):
    """Copy static folders, uuid, pyexiv2; optionally zip."""
    import sys
    platform = PLATFORM_MAPPING[sys.platform]

    source_root = os.path.abspath(source_root or os.getcwd())
    compile_path = os.path.abspath(compile_path)
    tgt_base = os.path.join(compile_path, PROJECT_NAME) if not onefile_mode else compile_path

    print("Copying static folders:")
    for src_folder in ["config", "weights", "global"]:
        if os.path.exists(os.path.join(source_root, src_folder)):
            copy_tree(src_folder, tgt_base, source_root)

    # uuid module
    import uuid
    if uuid.__file__:
        shutil.copy(uuid.__file__, tgt_base)

    # zip
    if apply_zip:
        zip_fname = os.path.join(
            compile_path, f"MetDetPy_{platform}_{VERSION}.zip")
        print(f"Zipping to {zip_fname}...", end="", flush=True)
        package_path = (compile_path if onefile_mode else os.path.join(
            compile_path, PROJECT_NAME))
        with zipfile.ZipFile(zip_fname,
                             mode='w',
                             compression=zipfile.ZIP_DEFLATED) as zf:
            file_to_zip(package_path, zf, archive_root=compile_path)
        print("Done.")
