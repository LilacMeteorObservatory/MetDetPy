# macOS 打包产物的动态库路径修复。
#
# 问题背景：
# 在 conda 环境下用 Nuitka 打包时，主程序的 LC_LOAD_DYLIB 会被写成
# 构建机的绝对路径（例如 /Users/<user>/anaconda3/envs/MetDetPy/lib/libpython3.10.dylib）。
# 该路径只存在于打包机上，产物拷贝到其它机器后 dyld 直接报
# "Library not loaded: /Users/<user>/anaconda3/..."。
#
# 同时，部分被收集进来的第三方 dylib（cv2 / av / rawpy 等 wheel 自带，
# 以及 Homebrew 的 brotli / gettext / inih）其 LC_ID_DYLIB 同样是构建机的
# 绝对路径（/DLC/cv2/.dylibs/... 、/opt/homebrew/opt/...），
# 属于构建机路径泄漏，同样需要改写。
#
# 修复方式：把这些绝对路径改写为相对于产物的路径
#   - LC_LOAD_DYLIB -> @loader_path/<相对路径>
#   - LC_ID_DYLIB   -> 与产物内既有约定一致的相对名（根目录用裸文件名）
# 同时把 ELF 风格的 $ORIGIN rpath 换成 macOS 原生的 @loader_path。
#
# 这里自己解析 Mach-O，而不是依赖 otool / install_name_tool 的输出格式，
# 便于在非 macOS 平台上做单元测试；实际改写仍由 install_name_tool 完成。
#
# 另外 install_name_tool 会让被改写二进制的代码签名失效（arm64 上表现为
# "invalid signature" 或 "not signed at all"），因此改写后需要用 ad-hoc
# 签名重新签一遍，恢复 Nuitka 原始产物的签名状态。

import os
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Mach-O magic
MH_MAGIC = 0xFEEDFACE
MH_CIGAM = 0xCEFAEDFE
MH_MAGIC_64 = 0xFEEDFACF
MH_CIGAM_64 = 0xCFFAEDFE
FAT_MAGIC = 0xCAFEBABE
FAT_CIGAM = 0xBEBAFECA
FAT_MAGIC_64 = 0xCAFEBABF
FAT_CIGAM_64 = 0xBFBAFECA

# Load command types
LC_REQ_DYLD = 0x80000000
LC_LOAD_DYLIB = 0x0C
LC_ID_DYLIB = 0x0D
LC_LOAD_WEAK_DYLIB = 0x18 | LC_REQ_DYLD
LC_LAZY_LOAD_DYLIB = 0x20
LC_REEXPORT_DYLIB = 0x1F | LC_REQ_DYLD
LC_LOAD_UPWARD_DYLIB = 0x23 | LC_REQ_DYLD
LC_RPATH = 0x1C | LC_REQ_DYLD

DYLIB_COMMANDS = (
    LC_LOAD_DYLIB,
    LC_ID_DYLIB,
    LC_LOAD_WEAK_DYLIB,
    LC_LAZY_LOAD_DYLIB,
    LC_REEXPORT_DYLIB,
    LC_LOAD_UPWARD_DYLIB,
)

# 系统库路径前缀，这些属于操作系统，不应被改写。
SYSTEM_PREFIXES = ("/usr/lib/", "/System/")

# ELF 风格的 rpath，macOS 的 dyld 并不识别。
ELF_ORIGIN_RPATH = "$ORIGIN"


@dataclass
class MachOInfo:
    """一个 Mach-O 文件中与动态库相关的 load command 信息。

    Attributes:
        load_dylibs: LC_LOAD_DYLIB 等引用的库路径。
        id_dylib: LC_ID_DYLIB，即该文件自身的 install name。
        rpaths: LC_RPATH 中记录的搜索路径。
    """

    load_dylibs: List[str] = field(default_factory=list)
    id_dylib: Optional[str] = None
    rpaths: List[str] = field(default_factory=list)


@dataclass
class Relocation:
    """一条待执行的 install name 改写。

    Attributes:
        kind: 改写类型，"change"（依赖路径）或 "id"（自身 install name）。
        old: 改写前的值。
        new: 改写后的值。
    """

    kind: str
    old: str
    new: str


def _is_system_path(path: str) -> bool:
    """判断是否为系统提供的库路径，这类路径不应被改写。"""
    return path.startswith(SYSTEM_PREFIXES)


def is_host_absolute_path(path: str) -> bool:
    """判断是否为构建机上的绝对路径（系统库除外）。

    Args:
        path (str): 待判断的库路径。

    Returns:
        bool: 是构建机绝对路径时返回 True。
    """
    return path.startswith("/") and not _is_system_path(path)


def _parse_thin_macho(data: bytes) -> MachOInfo:
    """解析单个架构的 Mach-O 数据。

    Args:
        data (bytes): Mach-O 数据片段。

    Returns:
        MachOInfo: 解析出的动态库信息。
    """
    info = MachOInfo()

    if len(data) < 28:
        return info

    magic = struct.unpack_from("<I", data, 0)[0]
    if magic in (MH_MAGIC_64, MH_CIGAM_64):
        endian = "<" if magic == MH_MAGIC_64 else ">"
        header_size = 32
    elif magic in (MH_MAGIC, MH_CIGAM):
        endian = "<" if magic == MH_MAGIC else ">"
        header_size = 28
    else:
        return info

    ncmds = struct.unpack_from(endian + "I", data, 16)[0]

    offset = header_size
    for _ in range(ncmds):
        if offset + 8 > len(data):
            break
        cmd, cmdsize = struct.unpack_from(endian + "II", data, offset)
        if cmdsize < 8:
            break

        if cmd in DYLIB_COMMANDS:
            name_offset = struct.unpack_from(endian + "I", data, offset + 8)[0]
            name = _read_cstring(data, offset + name_offset)
            if cmd == LC_ID_DYLIB:
                info.id_dylib = name
            else:
                info.load_dylibs.append(name)
        elif cmd == LC_RPATH:
            path_offset = struct.unpack_from(endian + "I", data, offset + 8)[0]
            info.rpaths.append(_read_cstring(data, offset + path_offset))

        offset += cmdsize

    return info


def _read_cstring(data: bytes, offset: int) -> str:
    """从字节数据中读取以 \\0 结尾的字符串。

    Args:
        data (bytes): 字节数据。
        offset (int): 字符串起始偏移。

    Returns:
        str: 读取到的字符串。
    """
    if offset >= len(data):
        return ""
    end = data.find(b"\0", offset)
    if end == -1:
        end = len(data)
    return data[offset:end].decode("utf-8", errors="replace")


def parse_macho(file_path: Path) -> MachOInfo:
    """解析 Mach-O 文件，提取其动态库相关信息。

    兼容 fat binary（多个架构）以及单架构文件。

    Args:
        file_path (Path): 待解析的文件路径。

    Returns:
        MachOInfo: 解析出的动态库信息。文件不是 Mach-O 时返回空结果。
    """
    try:
        data = file_path.read_bytes()
    except OSError:
        return MachOInfo()

    if len(data) < 4:
        return MachOInfo()

    magic = struct.unpack_from(">I", data, 0)[0]

    if magic in (FAT_MAGIC, FAT_MAGIC_64, FAT_CIGAM, FAT_CIGAM_64):
        info = MachOInfo()
        nfat_arch = struct.unpack_from(">I", data, 4)[0]
        entry_size = 20 if magic in (FAT_MAGIC, FAT_CIGAM) else 32
        for index in range(nfat_arch):
            entry_offset = 8 + index * entry_size
            if entry_offset + entry_size > len(data):
                break
            if magic in (FAT_MAGIC, FAT_CIGAM):
                _, _, slice_offset, slice_size, _ = struct.unpack_from(
                    ">IIIII", data, entry_offset)
            else:
                _, _, slice_offset, slice_size, _, _ = struct.unpack_from(
                    ">IIQQII", data, entry_offset)
            slice_info = _parse_thin_macho(
                data[slice_offset:slice_offset + slice_size])
            for name in slice_info.load_dylibs:
                if name not in info.load_dylibs:
                    info.load_dylibs.append(name)
            if info.id_dylib is None:
                info.id_dylib = slice_info.id_dylib
            for rpath in slice_info.rpaths:
                if rpath not in info.rpaths:
                    info.rpaths.append(rpath)
        return info

    return _parse_thin_macho(data)


def _collect_macho_files(dist_dir: Path) -> List[Path]:
    """收集产物目录下的所有 Mach-O 文件。

    Args:
        dist_dir (Path): 产物目录。

    Returns:
        List[Path]: Mach-O 文件列表。
    """
    result: List[Path] = []
    for path in sorted(dist_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                head = handle.read(4)
        except OSError:
            continue
        if len(head) < 4:
            continue
        if struct.unpack_from(">I", head, 0)[0] in (
                MH_MAGIC, MH_CIGAM, MH_MAGIC_64, MH_CIGAM_64, FAT_MAGIC,
                FAT_CIGAM, FAT_MAGIC_64, FAT_CIGAM_64):
            result.append(path)
    return result


def _build_basename_index(macho_files: Sequence[Path]) -> Dict[str, List[Path]]:
    """建立 文件名 -> 文件列表 的索引，用于定位产物内的库。

    Args:
        macho_files (Sequence[Path]): Mach-O 文件列表。

    Returns:
        Dict[str, List[Path]]: 文件名索引。
    """
    index: Dict[str, List[Path]] = {}
    for path in macho_files:
        index.setdefault(path.name, []).append(path)
    return index


def _resolve_target(reference: str, source: Path,
                    index: Dict[str, List[Path]]) -> Optional[Path]:
    """在产物内定位某个库文件。

    优先级：与引用方同目录 > 产物根目录 > 唯一同名文件。

    Args:
        reference (str): 原始的库路径。
        source (Path): 引用该库的 Mach-O 文件。
        index (Dict[str, List[Path]]): 文件名索引。

    Returns:
        Optional[Path]: 定位到的文件，未找到时返回 None。
    """
    candidates = index.get(os.path.basename(reference))
    if not candidates:
        return None

    for candidate in candidates:
        if candidate.parent == source.parent:
            return candidate

    if len(candidates) == 1:
        return candidates[0]

    # 同名文件有多个时，只在能唯一定位的情况下才返回，避免改写到
    # 错误的副本上（例如根目录与 pyexiv2/lib 下各有一份 libexiv2.dylib）。
    return None


def _relative_install_name(source: Path, target: Path) -> str:
    """计算从引用方到目标库的相对 install name。

    Args:
        source (Path): 引用方 Mach-O 文件。
        target (Path): 目标库文件。

    Returns:
        str: 形如 @loader_path/xxx 的相对路径。
    """
    relative = os.path.relpath(target, source.parent)
    return "@loader_path/" + relative.replace(os.sep, "/")


def plan_relocations(macho_files: Sequence[Path],
                     dist_dir: Path) -> Dict[Path, List[Relocation]]:
    """规划所有需要执行的 install name 改写。

    Args:
        macho_files (Sequence[Path]): 待处理的 Mach-O 文件列表。
        dist_dir (Path): 产物根目录。

    Returns:
        Dict[Path, List[Relocation]]: 文件到改写项的映射。
    """
    index = _build_basename_index(macho_files)
    plan: Dict[Path, List[Relocation]] = {}

    for path in macho_files:
        info = parse_macho(path)
        changes: List[Relocation] = []

        for reference in info.load_dylibs:
            if not is_host_absolute_path(reference):
                continue
            target = _resolve_target(reference, path, index)
            if target is None:
                continue
            changes.append(
                Relocation(kind="change",
                           old=reference,
                           new=_relative_install_name(path, target)))

        if info.id_dylib and is_host_absolute_path(info.id_dylib):
            # 产物根目录下的库沿用既有约定（裸文件名），子目录下的库使用
            # @loader_path 相对自身位置。
            if len(path.relative_to(dist_dir).parts) == 1:
                new_id = path.name
            else:
                new_id = "@loader_path/" + path.name
            changes.append(
                Relocation(kind="id", old=info.id_dylib, new=new_id))

        for rpath in info.rpaths:
            if rpath == ELF_ORIGIN_RPATH:
                changes.append(
                    Relocation(kind="rpath",
                               old=ELF_ORIGIN_RPATH,
                               new="@loader_path"))

        if changes:
            plan[path] = changes

    return plan


def _build_command(path: Path,
                   changes: Sequence[Relocation]) -> List[str]:
    """构造 install_name_tool 命令行。

    Args:
        path (Path): 目标 Mach-O 文件。
        changes (Sequence[Relocation]): 改写项。

    Returns:
        List[str]: 完整命令。
    """
    command = ["install_name_tool"]

    for change in changes:
        if change.kind == "change":
            command += ["-change", change.old, change.new]
        elif change.kind == "id":
            command += ["-id", change.new]
        elif change.kind == "rpath":
            command += ["-rpath", change.old, change.new]

    command.append(str(path))

    return command


def _is_signed(path: Path) -> bool:
    """判断文件当前是否带有有效的代码签名。

    Args:
        path (Path): Mach-O 文件。

    Returns:
        bool: 签名有效时返回 True。
    """
    codesign = shutil.which("codesign")
    if codesign is None:
        return False

    result = subprocess.run(["codesign", "-v", str(path)],
                            capture_output=True,
                            text=True)
    return result.returncode == 0


def _read_identifier(path: Path) -> Optional[str]:
    """读取二进制当前的代码签名 identifier。

    Args:
        path (Path): Mach-O 文件。

    Returns:
        Optional[str]: identifier；读不到时返回 None。
    """
    result = subprocess.run(["codesign", "-dv", str(path)],
                            capture_output=True,
                            text=True)
    for line in result.stderr.splitlines():
        if line.startswith("Identifier="):
            return line.split("=", 1)[1].strip()
    return None


def _resign_adhoc(path: Path, identifier: Optional[str] = None):
    """用 ad-hoc 签名恢复被 install_name_tool 破坏的代码签名。

    install_name_tool 改写 load command 后会作废原有的代码签名。产物在
    arm64 上必须带有效签名才能执行，因此改写完必须补签一次。

    Args:
        path (Path): 被改写的 Mach-O 文件。
        identifier (Optional[str]): 改写前的签名 identifier，补签时原样
            保留；为 None 时沿用 codesign 的默认推导。

    Raises:
        RuntimeError: codesign 执行失败。
    """
    codesign = shutil.which("codesign")
    if codesign is None:
        return

    command = [codesign, "-s", "-", "--force"]
    if identifier:
        command += ["-i", identifier]
    command.append(str(path))

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"codesign failed for '{path}': {result.stderr.strip()}")


def _apply_relocations(path: Path,
                        changes: Sequence[Relocation]) -> List[str]:
    """对单个文件执行 install name 改写。

    Args:
        path (Path): 目标 Mach-O 文件。
        changes (Sequence[Relocation]): 改写项。

    Returns:
        List[str]: 实际执行的命令，用于日志输出。
    """
    command = _build_command(path, changes)

    # 改写前先记下签名状态与 identifier：install_name_tool 会作废签名，
    # 原本带签名的产物必须补签回来，原本就没有签名的则保持原样。
    was_signed = _is_signed(path)
    identifier = _read_identifier(path) if was_signed else None

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"install_name_tool failed for '{path}': {result.stderr.strip()}")

    # install_name_tool 会作废原有签名，补签回来。
    if was_signed:
        _resign_adhoc(path, identifier)

    return [" ".join(_quote(part) for part in command)]


def _quote(text: str) -> str:
    """对命令行参数做 shell 引用。

    Args:
        text (str): 原始参数。

    Returns:
        str: 引用后的参数。
    """
    if not text:
        return "''"
    if all(char.isalnum() or char in "@%+=:,./-_$" for char in text):
        return text
    return "'" + text.replace("'", "'\\''") + "'"


def relocate_macos_binaries(dist_dir: str,
                            dry_run: bool = False) -> List[str]:
    """修复产物中的构建机绝对路径，使其可以在其它机器上运行。

    Args:
        dist_dir (str): 产物目录。
        dry_run (bool): 仅打印将要执行的命令，不实际改写。

    Returns:
        List[str]: 执行的（或将要执行的）命令列表。
    """
    root = Path(dist_dir).resolve()
    if not root.is_dir():
        return []

    macho_files = _collect_macho_files(root)
    plan = plan_relocations(macho_files, root)
    if not plan:
        return []

    commands: List[str] = []
    for path, changes in plan.items():
        if dry_run:
            command = _build_command(path, changes)
            commands.append(" ".join(_quote(part) for part in command))
            continue

        commands.extend(_apply_relocations(path, changes))

    return commands
