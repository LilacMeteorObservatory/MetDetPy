# Shared utilities for packaging backends.

import hashlib
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

from MetLib.utils import PROJECT_NAME, VERSION, PLATFORM_MAPPING


def run_cmd(command: list[str], env: Optional[dict[str, str]] = None):
    print("Running:", command)
    run_env = None
    if env:
        run_env = os.environ.copy()
        run_env.update(env)
    t_start = time.time()
    ret = subprocess.run(command, env=run_env)
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


def clean_output_dir(compile_path: str):
    """Remove a previous dist directory before a fresh build."""
    if os.path.isdir(compile_path):
        print(f"Cleaning output directory {compile_path} ...")
        shutil.rmtree(compile_path)


def post_process(compile_path: str,
                 onefile_mode: bool,
                 apply_zip: bool,
                 source_root: Optional[str] = None,
                 zip_name: Optional[str] = None) -> Optional[str]:
    """Copy static resource folders; optionally zip the artifacts.

    Returns the path of the generated ZIP archive, if any.
    """
    platform = PLATFORM_MAPPING[sys.platform]

    source_root = os.path.abspath(source_root or os.getcwd())
    compile_path = os.path.abspath(compile_path)
    tgt_base = os.path.join(compile_path, PROJECT_NAME) if not onefile_mode else compile_path

    print("Copying static folders:")
    for src_folder in ["config", "weights", "global"]:
        if os.path.exists(os.path.join(source_root, src_folder)):
            copy_tree(src_folder, tgt_base, source_root)

    # zip
    if not apply_zip:
        return None

    zip_fname = os.path.join(
        compile_path, zip_name or f"MetDetPy_{platform}_{VERSION}.zip")
    print(f"Zipping to {zip_fname}...", end="", flush=True)
    package_path = (compile_path if onefile_mode else os.path.join(
        compile_path, PROJECT_NAME))
    with zipfile.ZipFile(zip_fname,
                         mode='w',
                         compression=zipfile.ZIP_DEFLATED) as zf:
        file_to_zip(package_path, zf, archive_root=compile_path)
    print("Done.")
    return zip_fname


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_quiet(command: list[str]) -> Optional[str]:
    """Run a command and return its trimmed stdout, or None if it failed.

    ``None`` keeps a failed query distinguishable from a successful query with
    empty output, so the build report never reports a clean worktree when git
    itself could not run.
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def write_build_report(source_root: str,
                       compile_path: str,
                       onefile_mode: bool,
                       zip_path: Optional[str] = None) -> str:
    """Write the optional, minimal build report.

    Records only the build version, source commit/worktree state, toolchain
    versions and the SHA256 of every produced artifact. Enabled with
    ``make_package.py --build-report``.
    """
    source_root = os.path.abspath(source_root)
    compile_path = os.path.abspath(compile_path)

    commit = _run_quiet(["git", "-C", source_root, "rev-parse", "HEAD"])
    dirty = _run_quiet(["git", "-C", source_root, "status", "--porcelain"])
    if dirty is None:
        worktree_state = "未知（git status 查询失败）"
    else:
        worktree_state = "有未提交改动" if dirty else "无未提交改动"

    nuitka_lines = (
        _run_quiet([sys.executable, "-m", "nuitka", "--version"]) or "").splitlines()
    compiler = "未知"
    for line in nuitka_lines:
        if line.startswith("Version C compiler:"):
            compiler = line.split(":", 1)[1].strip()
            break

    lines = [
        f"{PROJECT_NAME} {VERSION} 构建报告",
        "=" * 60,
        f"版本      : {VERSION}",
        f"源码提交  : {commit or '未知（git rev-parse 查询失败）'}",
        f"工作区    : {worktree_state}",
        f"Python    : {_run_quiet([sys.executable, '--version']) or '未知'}",
        f"Nuitka    : {nuitka_lines[0] if nuitka_lines else '未知'}",
        f"C 编译器  : {compiler}",
        "",
        "构建产物：",
    ]

    artifacts: list[str] = []
    if os.path.isdir(compile_path):
        if onefile_mode:
            candidates = [os.path.join(compile_path, name)
                          for name in sorted(os.listdir(compile_path))]
        else:
            candidates = []
            package_dir = os.path.join(compile_path, PROJECT_NAME)
            for dirpath, _, filenames in os.walk(package_dir):
                candidates.extend(os.path.join(dirpath, name)
                                  for name in sorted(filenames))
        for path in candidates:
            if os.path.isfile(path) and path.lower().endswith((".exe", ".bin")):
                artifacts.append(path)
    for path in artifacts:
        lines.append(f"  {os.path.relpath(path, compile_path)}")
        lines.append(f"    SHA256 : {file_sha256(path)}")
    if zip_path and os.path.isfile(zip_path):
        lines.append(f"  {os.path.basename(zip_path)}")
        lines.append(f"    SHA256 : {file_sha256(zip_path)}")
    lines.append("")

    report_path = os.path.join(source_root, "build_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Build report written to {report_path}")
    return report_path
