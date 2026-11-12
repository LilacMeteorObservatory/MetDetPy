# Nuitka packaging backend.

import importlib.util
import os
import platform as pf
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

from MetLib.utils import PROJECT_NAME, VERSION, PLATFORM_MAPPING
from .common import (clean_output_dir, post_process, run_cmd,
                     write_build_report)

join_path = os.path.join

SCRIPTS = ["MetDetPy.py", "ClipToolkit.py", "MetDetPhoto.py"]

EXCLUDE_PKGS = [
    "torch", "torchvision", "scipy", "tensorflow", "keras", "IPython",
    "matplotlib", "pandas", "sympy", "PIL",
    "PySide2", "PySide6", "PyQt5", "PyQt6", "tkinter",
]

# Fixed onefile extraction cache: {CACHE_DIR}/<Product>/<cache tag>.
# The first launch of any onefile program unpacks its payload there; later
# launches reuse the unpacked files (Nuitka checks each file's CRC32), which
# removes the per-launch extraction overhead of the default
# {TEMP}/onefile_<PID>_<TIME> directory.
ONEFILE_CACHE_PRODUCT = PROJECT_NAME

# Windows onefile bootstrap patch for MSIX/APPX package virtualization.
# The patch lives in the repository (nuitka-patches/) and is applied to the
# build environment's Nuitka before compiling; it must not be silently
# skipped, otherwise the Windows Store build dies with 0xC0000135 before
# Python starts. See nuitka-patches/README.md.
MSIX_PATCH_SCRIPT = join_path("nuitka-patches", "patch_onefile_bootstrap.py")


def resolve_cache_tag(args) -> str:
    """Cache directory tag used in the onefile extraction path.

    Defaults to ``VERSION`` so every release gets its own fresh extraction
    cache. ``--build-tag`` is only needed when several builds of the *same*
    version must not share a cache.
    """
    tag = getattr(args, "build_tag", None) or VERSION
    # The tag becomes a path component; keep it filesystem-safe.
    for separator in ("/", "\\", ":", "*", "?", '"', "<", ">", "|"):
        tag = tag.replace(separator, "-")
    return tag


def onefile_cache_spec(cache_tag: str) -> str:
    """Nuitka ``--onefile-tempdir-spec`` value for the fixed cache path."""
    return "{CACHE_DIR}/" + ONEFILE_CACHE_PRODUCT + "/" + cache_tag


def display_cache_path(cache_spec: str) -> str:
    """Human readable form of ``cache_spec`` for console output."""
    if sys.platform == "win32":
        return cache_spec.replace("{CACHE_DIR}", "%LOCALAPPDATA%").replace(
            "/", "\\")
    # Nuitka's {CACHE_DIR} is $XDG_CACHE_HOME or ~/.cache on all non-Windows
    # platforms (macOS included), not ~/Library/Caches.
    return cache_spec.replace("{CACHE_DIR}", "~/.cache")


def _file_contains(path: str, needle: bytes,
                   chunk_size: int = 8 * 1024 * 1024) -> bool:
    """Return True if ``needle`` occurs in ``path`` (chunked read)."""
    overlap = max(len(needle) - 1, 0)
    tail = b""
    with open(path, "rb") as handle:
        while True:
            data = handle.read(chunk_size)
            if not data:
                return False
            if needle in tail + data:
                return True
            tail = (tail + data)[-overlap:] if overlap else b""


def load_msix_patch_module(work_path: str):
    """Load ``nuitka-patches/patch_onefile_bootstrap.py`` as a module.

    The patch script owns the marker constants (they are injected into the C
    source and checked in the produced executables), so loading it here keeps
    a single definition of the patch markers.
    """
    script = join_path(work_path, MSIX_PATCH_SCRIPT)
    if not os.path.isfile(script):
        sys.exit(f"ERROR: missing Nuitka MSIX patch script: {script}")
    spec = importlib.util.spec_from_file_location(
        "metdetpy_nuitka_onefile_patch", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_msix_onefile_patch(work_path: str):
    """Patch the build environment's Nuitka onefile bootstrap for MSIX/APPX.

    The patch script locks the Nuitka version and its own anchors, so a
    mismatch aborts the build instead of silently producing an engine that
    dies with 0xC0000135 in the Windows Store. Returns the loaded patch module
    (the caller needs its binary marker for the post-build verification).
    """
    patch_module = load_msix_patch_module(work_path)

    print("=" * 70)
    print("Applying Nuitka onefile MSIX/APPX bootstrap patch")
    print("=" * 70)
    ret_code, _ = run_cmd([sys.executable, patch_module.__file__])
    if ret_code != 0:
        sys.exit(
            f"ERROR: Nuitka MSIX patch failed (exit code {ret_code}); the "
            "Windows Store build would crash with 0xC0000135. Build aborted.")
    print()
    return patch_module


def verify_msix_onefile_patch(compile_path: str, marker: bytes) -> None:
    """Fail the build if any onefile executable lacks the bootstrap patch."""
    missing = []
    for script in SCRIPTS:
        exe = join_path(compile_path, script.replace(".py", ".exe"))
        if not os.path.isfile(exe) or not _file_contains(exe, marker):
            missing.append(os.path.basename(exe))
    if missing:
        sys.exit(
            "ERROR: Nuitka onefile MSIX patch marker missing in: "
            f"{', '.join(missing)}. The Windows Store build would crash with "
            "0xC0000135; refusing to finish the build.")
    print("Verified MSIX bootstrap patch marker in all onefile executables.")


def prepare_pyexiv2_onefile_support(
) -> tuple[list[str], dict[str, str], Optional[str]]:
    """Make pyexiv2 importable inside a Nuitka onefile payload.

    ``pyexiv2/lib/__init__.py`` loads ``exiv2.dll`` (or the platform
    equivalent) from its own directory and then does a bare
    ``import exiv2api``. In onefile mode the shared library has to be placed
    at ``pyexiv2/lib`` inside the payload, and the bare import only resolves
    if Nuitka knew the extension module at build time. Stage ``exiv2api`` so
    ``--include-module`` can find and register it.

    Returns (extra Nuitka arguments, extra build environment, staging dir).
    """
    extra_args: list[str] = []
    extra_env: dict[str, str] = {}
    stage_dir: Optional[str] = None

    spec = importlib.util.find_spec("pyexiv2")
    if spec is None or not spec.submodule_search_locations:
        print("WARNING: pyexiv2 is not installed in the build environment; "
              "EXIF/ICC writing will fall back to plain output.")
        return extra_args, extra_env, stage_dir

    native_dir = Path(list(spec.submodule_search_locations)[0]) / "lib"
    if not native_dir.is_dir():
        print(f"WARNING: pyexiv2 native directory not found at {native_dir}.")
        return extra_args, extra_env, stage_dir

    extension: Optional[Path] = None
    for suffix in (".pyd", ".so", ".dylib"):
        candidate = native_dir / f"exiv2api{suffix}"
        if candidate.is_file():
            extension = candidate
            break

    shared: Optional[Path] = None
    for name in ("exiv2.dll", "libexiv2.so", "libexiv2.dylib"):
        candidate = native_dir / name
        if candidate.is_file():
            shared = candidate
            break

    if extension is None:
        print("WARNING: pyexiv2 native extension (exiv2api) not found; "
              "EXIF/ICC writing will fall back to plain output.")
        return extra_args, extra_env, stage_dir

    stage_dir = tempfile.mkdtemp(prefix="metdetpy-pyexiv2-")
    try:
        shutil.copy2(extension, Path(stage_dir) / extension.name)
    except OSError:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    extra_args.append("--include-module=exiv2api")
    if shared is not None:
        # Onefile extraction keeps the relative target path inside the
        # payload, which is exactly where pyexiv2 looks for the library.
        extra_args.append(f"--include-data-files={shared}=pyexiv2/lib/{shared.name}")
    else:
        print("WARNING: pyexiv2 shared library not found; EXIF/ICC writing may "
              "fall back to plain output.")

    python_path = stage_dir
    if os.environ.get("PYTHONPATH"):
        python_path += os.pathsep + os.environ["PYTHONPATH"]
    extra_env["PYTHONPATH"] = python_path
    return extra_args, extra_env, stage_dir


def expose_pyexiv2_extension(dist_dir: str):
    # pyexiv2.lib imports the extension by its top-level name, exiv2api.
    # Nuitka collects it under pyexiv2/lib; expose it on the bundle's sys.path.
    bundle = Path(dist_dir)
    native_dir = bundle / "pyexiv2" / "lib"
    for extension in native_dir.glob("exiv2api.*"):
        if extension.suffix in (".pyd", ".so"):
            shutil.copy2(extension, bundle / extension.name)
    # Its ctypes loader also expects the shared library beside __file__.
    for name in ("exiv2.dll", "libexiv2.so", "libexiv2.dylib"):
        library = bundle / name
        if native_dir.is_dir() and library.is_file():
            shutil.copy2(library, native_dir / name)


def nuitka_compile(header: list[str], options: dict[str, Union[bool, str]],
                   nuitka_pkgs: list[str], target: str,
                   env: Optional[dict[str, str]] = None):
    options_list = [
        key if value is True else f'{key}={value}'
        for key, value in options.items() if value
    ]
    merged = header + options_list + nuitka_pkgs + [target]

    ret_code, time_cost = run_cmd(merged, env=env)
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

    cache_tag = resolve_cache_tag(args)
    cache_spec = onefile_cache_spec(cache_tag)

    print("Use nuitka as package tool.")

    if getattr(args, "clean_dist", False):
        clean_output_dir(compile_path)

    compile_tool = [sys.executable, "-m", "nuitka"]

    nuitka_base: dict[str, Union[bool, str]] = {
        "--no-pyi-file": True,
        "--remove-output": True,
        "--lto": "yes",
        "--assume-yes-for-downloads": True,
    }

    nuitka_pkgs = [f"--nofollow-import-to={x}" for x in EXCLUDE_PKGS]
    # PyAV extensions import other av modules dynamically (e.g. av.utils).
    nuitka_pkgs.append("--include-module=av.utils")
    nuitka_pkgs.append("--include-package-data=pyexiv2")

    if platform == "win":
        if sys.version_info >= (3, 13):
            if args.mingw64:
                print("WARNING: MinGW64 is not supported for Python 3.13+. "
                      "Using MSVC instead; install Visual Studio Build Tools "
                      "with the Desktop development with C++ workload.")
            nuitka_base["--msvc"] = "latest"
        elif args.mingw64:
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
    pyexiv2_pkgs: list[str] = []
    build_env: Optional[dict[str, str]] = None
    pyexiv2_stage: Optional[str] = None
    msix_patch_module = None

    # Both the pyexiv2 staging directory and the MSIX patch step must be
    # inside this scope: the staging directory has to be released even when
    # the patch (or any compilation) aborts the build.
    try:
        if onefile_mode:
            print(f"onefile extraction cache tag: {cache_tag}")
            print(f"onefile extraction cache: {cache_spec}")
            print(f"  -> resolves to {display_cache_path(cache_spec)}")
            print("The cache is keyed by VERSION; pass --build-tag only when "
                  "several builds of the same version must not share it.")
            pyexiv2_pkgs, build_env, pyexiv2_stage = \
                prepare_pyexiv2_onefile_support()

        if onefile_mode and platform == "win":
            # Must be applied before Nuitka compiles the onefile bootstrap.
            # The Windows Store (MSIX/APPX) build dies with 0xC0000135
            # otherwise.
            msix_patch_module = apply_msix_onefile_patch(work_path)

        for script in SCRIPTS:
            cfg: dict[str, Union[bool, str]] = {
                "--standalone": True,
                "--output-dir": compile_path,
            }
            cfg.update(nuitka_base)
            if onefile_mode:
                cfg["--onefile"] = True
                cfg["--onefile-tempdir-spec"] = cache_spec
                # A static spec already infers "cached"; set it explicitly so
                # a future Nuitka default change cannot silently reintroduce
                # per-launch extraction.
                cfg["--onefile-cache-mode"] = "cached"
                # LTO adds a lot of link time for three executables and does
                # not noticeably shrink them.
                cfg["--lto"] = "no"

            nuitka_compile(compile_tool, cfg, nuitka_pkgs + pyexiv2_pkgs,
                           target=join_path(work_path, script),
                           env=build_env)
    finally:
        if pyexiv2_stage is not None:
            shutil.rmtree(pyexiv2_stage, ignore_errors=True)

    if onefile_mode and platform == "win":
        # Hard gate: never publish an onefile build whose bootstrap lacks the
        # MSIX patch (that build cannot start in the Windows Store).
        if msix_patch_module is None:
            sys.exit("ERROR: Nuitka MSIX bootstrap patch was not applied; "
                     "refusing to finish the build.")
        verify_msix_onefile_patch(
            compile_path, msix_patch_module.BINARY_MARKER.encode("ascii"))

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
        expose_pyexiv2_extension(join_path(compile_path, "MetDetPy"))
        print("Done.")

    zip_name = None
    if args.apply_zip:
        # Release naming style like the GitHub release assets:
        # MetDetPy-win-v2.6.1-nuitka-onefile.zip / -nuitka-onedir.zip
        mode_tag = "onefile" if onefile_mode else "onedir"
        zip_name = (f"{PROJECT_NAME}-{platform}-"
                    f"{VERSION.lower()}-nuitka-{mode_tag}.zip")

    zip_path = post_process(compile_path,
                            onefile_mode,
                            args.apply_zip,
                            source_root=work_path,
                            zip_name=zip_name)

    if onefile_mode:
        print(f"onefile extraction cache: {display_cache_path(cache_spec)}")
        print("The cache is kept after exit and reused on later launches; "
              "delete it while all packaged programs are closed to force a "
              "fresh extraction.")

    if getattr(args, "build_report", False):
        write_build_report(work_path,
                           compile_path,
                           onefile_mode,
                           zip_path=zip_path)
