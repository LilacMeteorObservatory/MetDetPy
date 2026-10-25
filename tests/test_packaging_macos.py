"""Tests for the macOS dynamic library path relocation used by packaging.

The tests build synthetic Mach-O files so they can run on any platform,
without requiring a real toolchain or the actual packaged artifacts.
"""
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "MetLib" /
                       "packaging"))

import macos  # noqa: E402


MH_MAGIC_64 = 0xFEEDFACF
LC_ID_DYLIB = 0x0D
LC_LOAD_DYLIB = 0x0C
LC_RPATH = 0x1C | 0x80000000


def _align8(size: int) -> int:
    """Round a load command size up to the required 8-byte multiple."""
    return size + (-size % 8)


def _dylib_command(cmd: int, name: str) -> bytes:
    """Build a LC_LOAD_DYLIB / LC_ID_DYLIB load command."""
    name_bytes = name.encode("utf-8") + b"\0"
    cmdsize = _align8(24 + len(name_bytes))
    # cmd, cmdsize, name offset, timestamp, current version, compat version
    header = struct.pack("<IIIIII", cmd, cmdsize, 24, 0, 0, 0)
    body = header + name_bytes
    return body + b"\0" * (cmdsize - len(body))


def _rpath_command(path: str) -> bytes:
    """Build an LC_RPATH load command."""
    path_bytes = path.encode("utf-8") + b"\0"
    cmdsize = _align8(12 + len(path_bytes))
    # cmd, cmdsize, path offset
    header = struct.pack("<III", LC_RPATH, cmdsize, 12)
    body = header + path_bytes
    return body + b"\0" * (cmdsize - len(body))


def write_macho(path: Path,
                load_dylibs: list[str] = (),
                id_dylib: str = None,
                rpaths: list[str] = ()) -> Path:
    """Write a minimal single-architecture Mach-O file.

    Args:
        path (Path): Output path.
        load_dylibs (list[str]): LC_LOAD_DYLIB entries.
        id_dylib (str): LC_ID_DYLIB entry.
        rpaths (list[str]): LC_RPATH entries.

    Returns:
        Path: The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    command_blobs = []
    for name in load_dylibs:
        command_blobs.append(_dylib_command(LC_LOAD_DYLIB, name))
    if id_dylib is not None:
        command_blobs.append(_dylib_command(LC_ID_DYLIB, id_dylib))
    for rpath in rpaths:
        command_blobs.append(_rpath_command(rpath))

    commands = b"".join(command_blobs)

    # 64-bit Mach-O header: magic, cputype, cpusubtype, filetype, ncmds,
    # sizeofcmds, flags, reserved
    header = struct.pack("<IiiIIIII", MH_MAGIC_64, 0x0100000C, 0, 2,
                          len(command_blobs), len(commands), 0, 0)
    # pad header to the 64-bit size (32 bytes)
    header = header + b"\0" * (32 - len(header))

    path.write_bytes(header + commands)
    return path


def test_parse_macho_reads_load_commands(tmp_path):
    target = write_macho(
        tmp_path / "main.bin",
        load_dylibs=["/build/env/lib/libpython3.10.dylib", "/usr/lib/libSystem.B.dylib"],
        id_dylib="main.bin",
        rpaths=["$ORIGIN"])

    info = macos.parse_macho(target)

    assert info.load_dylibs == [
        "/build/env/lib/libpython3.10.dylib",
        "/usr/lib/libSystem.B.dylib",
    ]
    assert info.id_dylib == "main.bin"
    assert info.rpaths == ["$ORIGIN"]


def test_parse_macho_ignores_non_macho(tmp_path):
    plain = tmp_path / "notes.txt"
    plain.write_bytes(b"just some text, definitely not a mach-o file")

    info = macos.parse_macho(plain)

    assert info.load_dylibs == []
    assert info.id_dylib is None


def test_is_host_absolute_path_ignores_system_libs():
    assert macos.is_host_absolute_path("/Users/dev/anaconda3/envs/X/lib/libpython3.10.dylib")
    assert macos.is_host_absolute_path("/opt/homebrew/opt/brotli/lib/libbrotli.dylib")
    # system libraries and relative references must be left untouched
    assert not macos.is_host_absolute_path("/usr/lib/libSystem.B.dylib")
    assert not macos.is_host_absolute_path("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    assert not macos.is_host_absolute_path("@executable_path/libssl.3.dylib")
    assert not macos.is_host_absolute_path("libpython3.10.dylib")


def test_plan_relocations_rewrites_absolute_libpython(tmp_path):
    write_macho(tmp_path / "libpython3.10.dylib", id_dylib="libpython3.10.dylib")
    main = write_macho(
        tmp_path / "main.bin",
        load_dylibs=["/build/env/lib/libpython3.10.dylib"],
        id_dylib="main.bin",
        rpaths=["$ORIGIN"])

    plan = macos.plan_relocations([main, tmp_path / "libpython3.10.dylib"], tmp_path)

    changes = plan[main]
    assert macos.Relocation("change", "/build/env/lib/libpython3.10.dylib",
                             "@loader_path/libpython3.10.dylib") in changes
    # the ELF-style $ORIGIN rpath is meaningless to the macOS dyld
    assert macos.Relocation("rpath", "$ORIGIN", "@loader_path") in changes


def test_plan_relocations_keeps_system_and_relative_refs(tmp_path):
    main = write_macho(
        tmp_path / "main.bin",
        load_dylibs=["/usr/lib/libSystem.B.dylib", "@executable_path/libssl.3.dylib"],
        id_dylib="main.bin")

    plan = macos.plan_relocations([main], tmp_path)

    assert main not in plan


def test_plan_relocations_rewrites_absolute_dylib_id(tmp_path):
    lib = write_macho(
        tmp_path / "libfoo.dylib",
        id_dylib="/opt/homebrew/opt/foo/lib/libfoo.dylib")

    plan = macos.plan_relocations([lib], tmp_path)

    assert macos.Relocation("id", "/opt/homebrew/opt/foo/lib/libfoo.dylib",
                             "libfoo.dylib") in plan[lib]


def test_plan_relocations_ignores_ambiguous_targets(tmp_path):
    # Two libraries share a basename and neither sits next to the referencing
    # binary, so the reference must not be rewritten to an arbitrary one.
    write_macho(tmp_path / "libfoo.dylib", id_dylib="libfoo.dylib")
    write_macho(tmp_path / "sub" / "libfoo.dylib", id_dylib="libfoo.dylib")
    main = write_macho(
        tmp_path / "other" / "main.bin",
        load_dylibs=["/opt/homebrew/lib/libfoo.dylib"],
        id_dylib="main.bin")

    plan = macos.plan_relocations(
        [main, tmp_path / "libfoo.dylib", tmp_path / "sub" / "libfoo.dylib"],
        tmp_path)

    assert main not in plan


def test_plan_relocations_resolves_same_directory_preferred(tmp_path):
    write_macho(tmp_path / "libfoo.dylib", id_dylib="libfoo.dylib")
    write_macho(tmp_path / "sub" / "libfoo.dylib", id_dylib="libfoo.dylib")
    main = write_macho(
        tmp_path / "sub" / "main.bin",
        load_dylibs=["/opt/homebrew/lib/libfoo.dylib"],
        id_dylib="main.bin")

    plan = macos.plan_relocations(
        [main, tmp_path / "libfoo.dylib", tmp_path / "sub" / "libfoo.dylib"],
        tmp_path)

    assert macos.Relocation("change", "/opt/homebrew/lib/libfoo.dylib",
                             "@loader_path/libfoo.dylib") in plan[main]


@pytest.mark.skipif(sys.platform != "darwin", reason="requires install_name_tool")
def test_relocate_macos_binaries_applies_and_verifies(tmp_path):
    write_macho(tmp_path / "libpython3.10.dylib", id_dylib="libpython3.10.dylib")
    main = write_macho(
        tmp_path / "main.bin",
        load_dylibs=["/build/env/lib/libpython3.10.dylib"],
        id_dylib="main.bin",
        rpaths=["$ORIGIN"])

    commands = macos.relocate_macos_binaries(str(tmp_path))

    assert commands

    info = macos.parse_macho(main)
    assert info.load_dylibs == ["@loader_path/libpython3.10.dylib"]
    assert info.rpaths == ["@loader_path"]


def test_apply_relocations_resigns_with_original_identifier(tmp_path,
                                                           monkeypatch):
    """install_name_tool 会作废签名，补签必须发生在改写之后并保留 identifier."""
    target = tmp_path / "main.bin"
    target.write_bytes(b"stub")

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return _Result()

    monkeypatch.setattr(macos.subprocess, "run", fake_run)
    monkeypatch.setattr(macos, "_read_identifier", lambda path: "org.example.app")
    monkeypatch.setattr(
        macos, "_resign_adhoc",
        lambda path, identifier=None: calls.append(("resign", identifier)))

    macos._apply_relocations(
        target, [macos.Relocation("id", "/old/lib.dylib", "lib.dylib")])

    # 补签发生在 install_name_tool 之后
    tool_calls = [c for c in calls
                  if isinstance(c, list) and c[0] == "install_name_tool"]
    assert tool_calls
    assert calls[-1] == ("resign", "org.example.app")


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="requires install_name_tool and codesign")
def test_relocation_restores_code_signature(tmp_path):
    """改写真实二进制后，代码签名必须仍然有效。"""
    import shutil
    import subprocess

    source = Path("/usr/bin/true")
    if not source.is_file():
        pytest.skip("/usr/bin/true not available")

    target = tmp_path / "true.bin"
    # copyfile 只复制内容：copy2 会连带复制 /usr/bin/true 的扩展属性与
    # 文件标志，在 SIP 保护的系统文件上会抛 PermissionError。
    shutil.copyfile(source, target)
    subprocess.run(["codesign", "-s", "-", "--force", str(target)],
                   capture_output=True, check=True)

    macos._apply_relocations(
        target, [macos.Relocation("id", "/old/lib.dylib", "lib.dylib")])

    verify = subprocess.run(["codesign", "-v", str(target)],
                            capture_output=True, text=True)
    assert verify.returncode == 0, verify.stderr

    detail = subprocess.run(["codesign", "-dv", str(target)],
                             capture_output=True, text=True)
    assert "Signature=adhoc" in detail.stderr
