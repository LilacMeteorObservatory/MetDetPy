# Shared utilities for packaging backends.

import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from MetLib.utils import PROJECT_NAME, VERSION, PLATFORM_MAPPING


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
    tgt_dir = os.path.join(tgt_path, tree_path)
    if os.path.exists(tgt_dir):
        print("exists, skipped.")
        return
    shutil.copytree(f"./{tree_path}", tgt_dir)
    print("ok.")


def post_process(compile_path: str, onefile_mode: bool, apply_zip: bool):
    """Copy static folders, uuid, pyexiv2; optionally zip."""
    import sys
    platform = PLATFORM_MAPPING[sys.platform]

    tgt_base = os.path.join(compile_path, PROJECT_NAME) if not onefile_mode else compile_path

    print("Copying static folders:")
    for src_folder in ["config", "weights", "global"]:
        if os.path.exists(src_folder):
            copy_tree(src_folder, tgt_base)

    # uuid module
    import uuid
    shutil.copy(uuid.__file__, tgt_base)

    # zip
    if apply_zip:
        zip_fname = os.path.join(
            compile_path, f"MetDetPy_{platform}_{VERSION}.zip")
        print(f"Zipping to {zip_fname}...", end="", flush=True)
        with zipfile.ZipFile(zip_fname, mode='w') as zf:
            file_to_zip(os.path.join(compile_path, PROJECT_NAME), zf)
        print("Done.")
