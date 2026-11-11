#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给 Nuitka 的 onefile Windows 引导层打 MSIX/APPX 包虚拟化补丁。

背景
====
Windows 商店版（APPX / MSIX）里，文件系统虚拟化会把打包应用写入
``%LOCALAPPDATA%`` 的路径重定向到包私有目录::

    逻辑路径 A = %LOCALAPPDATA%\\MetDetPy\\<CACHE_TAG>
    物理路径 B = %LOCALAPPDATA%\\Packages\\<PFN>\\LocalCache\\Local\\MetDetPy\\<CACHE_TAG>
    （<CACHE_TAG> 默认为 VERSION，可用 --build-tag 覆盖。）

onefile 引导层把载荷解压到 A（物理上落在 B），随后用 **逻辑路径** 启动内部
子进程。Windows 加载器于是把逻辑路径当作「应用目录」去查依赖 DLL
（``python310.dll``），而该文件只存在于 B —— 子进程在 Python 解释器加载之前
就以 ``0xC0000135``（STATUS_DLL_NOT_FOUND）退出，识别功能完全不可用。

这不是 MetDetPy 代码的问题（资源目录用 ``sys.argv[0]`` 定位，本身是对的），
而是 Nuitka onefile 引导层在 MSIX 下的路径选择问题。MeteorMasterDetecterAI
2.5 已经用同一条补丁修过（见 ``U:\\yolo\\MeteorMasterDetecterAI\\nuitka-patches``），
本文件是它的 MetDetPy 版本。

修法
====
在启动内部子进程之前，把逻辑路径解析为最终物理路径
（``GetFinalPathNameByHandleW``），用物理路径作为 ``CreateProcessW`` 的
``lpApplicationName``。这样子进程的应用目录指向真实目录，依赖 DLL 就在旁边。

约束
====
* 只在 Windows 生效；解析失败时回退到原逻辑路径，非商店环境行为不变。
* 锁定 Nuitka 版本 + 被改文件的预期内容，命中不上直接失败，避免静默漏补丁。
* 确定性重放：始终以 ``.metdetpy-orig`` 备份为真源重新打补丁，重复执行结果一致。
* **补丁注入的 C 代码必须是纯 ASCII**。原因见 ``assert_ascii`` 的注释。

用法
----
    python nuitka-patches/patch_onefile_bootstrap.py

``MetLib/packaging/nuitka.py`` 在 Windows onefile 构建前会自动调用本脚本。
"""

import os
import shutil
import sys

# 锁定的 Nuitka 版本。换了版本必须重新核对锚点，不允许「差不多就行」。
# 4.0.5 的三个锚点与 2.7.13 完全一致（各命中 1 次），只是版本号需要更新。
EXPECTED_NUITKA_VERSION = "4.0.5"

# 补丁标记：出现在被改文件里，用来识别「已经打过补丁」。
SOURCE_MARKER = "METDETPY_MSIX_ONEFILE_PATCH"

# 写进产物二进制的标记字符串，构建后由 nuitka.py 检查，防止静默漏打。
# （nuitka.py 直接加载本脚本读取该常量，整个仓库只有这一处定义。）
BINARY_MARKER = SOURCE_MARKER + "_ACTIVE"

# --- 锚点：命中不上就失败，绝不猜测 -------------------------------------------------

# 锚点 1：Windows 专用区段内的子进程命令行构造函数。
ANCHOR_HELPER = "static wchar_t *getCommandLineForChildProcess(void) {"

# 锚点 2：内部子进程可执行文件路径的选取。
ANCHOR_FORK = """#if _NUITKA_ONEFILE_DLL_MODE
    filename_char_t const *fork_binary = getBinaryPath();
#else
    filename_char_t const *fork_binary = first_filename;
#endif"""

# 锚点 3：真正启动子进程的地方（整块替换，便于在后面释放路径缓冲区）。
ANCHOR_CREATE = """    bool_res = CreateProcessW(fork_binary,
                              getCommandLineForChildProcess(), // command line
                              NULL,                            // process attributes
                              NULL,                            // thread attributes
                              TRUE,                            // inherit handles
                              NORMAL_PRIORITY_CLASS,           // creation flags
                              NULL, NULL, &si, &pi);"""

# --- 补丁内容（必须纯 ASCII，原因见 assert_ascii） ----------------------------------

HELPER_CODE = r'''// METDETPY_MSIX_ONEFILE_PATCH begin
// Resolve a logical path to its final physical path (Windows only).
//
// In the Windows Store (APPX/MSIX) build, the onefile cache directory is a
// logical path under %LOCALAPPDATA%. Package file system virtualization
// redirects it to a package private directory. The bootstrap unpacks the
// payload through that redirect (works fine), but the child process was then
// started with the *logical* path, so the Windows loader used the logical
// path as the application directory when looking for dependent DLLs such as
// python310.dll. That DLL only exists in the package private directory, so
// the child died with 0xC0000135 (STATUS_DLL_NOT_FOUND) before the Python
// interpreter was even loaded.
//
// Starting the child with the physical path makes the application directory
// point at the real directory, where the dependent DLLs live.
//
// dwDesiredAccess is 0 (query attributes only), sharing allows read/write/
// delete. Returns NULL on failure, the caller then falls back to the
// original logical path.
static wchar_t *getFinalPhysicalPathW(wchar_t const *logical_path) {
    HANDLE file_handle = CreateFileW(logical_path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                      NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file_handle == INVALID_HANDLE_VALUE) {
        return NULL;
    }

    // First call only asks for the required length (the value includes the
    // terminating null character).
    DWORD const needed = GetFinalPathNameByHandleW(file_handle, NULL, 0, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (needed == 0) {
        CloseHandle(file_handle);
        return NULL;
    }

    wchar_t *buffer = (wchar_t *)malloc((needed + 1) * sizeof(wchar_t));
    if (buffer == NULL) {
        CloseHandle(file_handle);
        return NULL;
    }

    DWORD const length =
        GetFinalPathNameByHandleW(file_handle, buffer, needed + 1, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    CloseHandle(file_handle);

    if (length == 0 || length > needed) {
        free(buffer);
        return NULL;
    }
    buffer[length] = 0;

    // Strip the Win32 extended path prefix:
    //   \\?\UNC\server\share\x -> \\server\share\x
    //   \\?\C:\dir\x          -> C:\dir\x
    if (wcsncmp(buffer, L"\\\\?\\UNC\\", 8) == 0) {
        memmove(buffer + 2, buffer + 8, (wcslen(buffer + 8) + 1) * sizeof(wchar_t));
    } else if (wcsncmp(buffer, L"\\\\?\\", 4) == 0) {
        memmove(buffer, buffer + 4, (wcslen(buffer + 4) + 1) * sizeof(wchar_t));
    }

    return buffer;
}
// METDETPY_MSIX_ONEFILE_PATCH end
'''

FORK_CODE = """#if _NUITKA_ONEFILE_DLL_MODE
    filename_char_t const *fork_binary = getBinaryPath();
#else
    filename_char_t const *fork_binary = first_filename;
#endif

// METDETPY_MSIX_ONEFILE_PATCH begin
// Resolve the real physical path of the child binary (MSIX/APPX package
// virtualization fix). In the Store build the logical path is redirected to
// a package private directory; starting the child with the physical path is
// what lets the Windows loader find the dependent DLLs (python310.dll) next
// to it. On failure the original logical path is kept, so non-Store builds
// behave exactly as before.
#if defined(_WIN32)
    wchar_t const *fork_binary_final = (wchar_t const *)fork_binary;
    wchar_t *fork_binary_physical = getFinalPhysicalPathW((wchar_t const *)fork_binary);
    if (fork_binary_physical != NULL) {
        fork_binary_final = fork_binary_physical;
    }
#endif
// METDETPY_MSIX_ONEFILE_PATCH end"""

CREATE_CODE = r"""    // METDETPY_MSIX_ONEFILE_PATCH begin
    // Diagnostics go to stderr and are flushed right away: stdout carries the
    // JSON line protocol with the detection results and must stay clean.
    fprintf(stderr,
            "[MetDetPy onefile] METDETPY_MSIX_ONEFILE_PATCH_ACTIVE fork target: logical=\"%ls\" final=\"%ls\"\n",
            (wchar_t const *)fork_binary, (wchar_t const *)fork_binary_final);
    fflush(stderr);

    bool_res = CreateProcessW(fork_binary_final,
                              getCommandLineForChildProcess(), // command line
                              NULL,                            // process attributes
                              NULL,                            // thread attributes
                              TRUE,                            // inherit handles
                              NORMAL_PRIORITY_CLASS,           // creation flags
                              NULL, NULL, &si, &pi);

    // The physical path buffer is not needed anymore, the child has its own
    // copy of the command line.
    if (fork_binary_physical != NULL) {
        free(fork_binary_physical);
        fork_binary_physical = NULL;
    }
    // METDETPY_MSIX_ONEFILE_PATCH end"""


def fail(message):
    sys.stderr.write("[nuitka-patch] FAIL: %s\n" % message)
    sys.exit(1)


def assert_ascii(block, name):
    """补丁注入的 C 代码必须是纯 ASCII。

    原因：MSVC 默认按系统代码页（本机是 936 / GBK）解读源文件。UTF-8 编码的
    中文字符，其尾字节会落在 DBCS 前导字节区间（0x81-0xFE）。如果某个中文注释
    行恰好以这样的字节结尾，MSVC 的 DBCS 解码器会把「前导字节 + 换行」当成一个
    非法双字节字符，**换行被吞掉**，下一行的 ``#endif`` 就被 ``//`` 注释吃掉，
    编译报 ``fatal error C1020: unterminated #endif``。

    这个坑非常隐蔽（报错行号指向中文注释，而不是真正出错的 #endif），所以在
    这里直接拦下来，不要交给编译器去发现。
    """
    for index, char in enumerate(block):
        if ord(char) > 0x7F:
            fail("%s 含非 ASCII 字符 %r（第 %d 个字符）。补丁注入的 C 代码必须是纯 ASCII。"
                 % (name, char, index))


def assert_no_dbsc_lead_at_eol(text, path):
    """兜底检查：任何一行都不能以 DBCS 前导字节结尾（会被 MSVC 吞掉换行）。"""
    lines = text.split("\n")
    offenders = []
    for number, line in enumerate(lines[:-1], start=1):
        if not line:
            continue
        encoded = line.encode("utf-8")
        last = encoded[-1]
        if 0x81 <= last <= 0xFE:
            offenders.append(number)
    if offenders:
        fail(
            "%s 的第 %s 行以 DBCS 前导字节结尾，在 936 代码页下会让 MSVC 吞掉换行"
            "（进而报 unterminated #endif）。请把这些行改成纯 ASCII。"
            % (path, ", ".join(str(item) for item in offenders[:10]))
        )


def get_nuitka_version():
    try:
        import importlib.metadata as metadata

        return metadata.version("Nuitka")
    except Exception:
        try:
            from nuitka.Version import getNuitkaVersion

            return getNuitkaVersion()
        except Exception:
            return None


def get_bootstrap_path():
    try:
        import nuitka
    except ImportError:
        fail("import nuitka 失败：请用打包环境的 Python 运行本脚本（%s）" % sys.executable)

    static_src = os.path.join(os.path.dirname(os.path.abspath(nuitka.__file__)), "build", "static_src")
    path = os.path.join(static_src, "OnefileBootstrap.c")
    if not os.path.isfile(path):
        fail("找不到 Nuitka 引导层源码：%s" % path)
    return path


def main():
    version = get_nuitka_version()
    if version != EXPECTED_NUITKA_VERSION:
        fail(
            "Nuitka 版本不匹配：当前 %s，补丁针对 %s。"
            "请先核对锚点是否仍然成立，再更新 EXPECTED_NUITKA_VERSION。"
            % (version, EXPECTED_NUITKA_VERSION)
        )

    assert_ascii(HELPER_CODE, "HELPER_CODE")
    assert_ascii(FORK_CODE, "FORK_CODE")
    assert_ascii(CREATE_CODE, "CREATE_CODE")

    if BINARY_MARKER not in CREATE_CODE:
        fail("CREATE_CODE 里缺少 BINARY_MARKER，构建后的二进制校验会失效。")

    path = get_bootstrap_path()
    backup = path + ".metdetpy-orig"

    # 确定性重放：始终以备份为真源。这样重复执行、或者上一轮打了半截补丁，
    # 结果都跟「原始文件 + 补丁」完全一致。
    if os.path.isfile(backup):
        with open(backup, "r", encoding="utf-8", newline="") as handle:
            text = handle.read()
        source = backup
    else:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            text = handle.read()
        shutil.copy2(path, backup)
        source = path
        print("[nuitka-patch] 已备份原始文件：%s" % backup)

    if SOURCE_MARKER in text:
        fail("备份文件 %s 里已经带补丁标记，说明它不是原始文件。请人工核对后再继续。" % backup)

    newline = "\r\n" if "\r\n" in text else "\n"

    def nl(value):
        return value.replace("\n", newline)

    anchors = (("ANCHOR_HELPER", ANCHOR_HELPER), ("ANCHOR_FORK", ANCHOR_FORK),
               ("ANCHOR_CREATE", ANCHOR_CREATE))
    for name, anchor in anchors:
        count = text.count(nl(anchor))
        if count != 1:
            fail("锚点 %s 在 %s 里命中 %d 次（必须恰好 1 次）。"
                 "Nuitka 版本或源码结构变了，需要人工重新定位补丁位置。"
                 % (name, source, count))

    text = text.replace(nl(ANCHOR_HELPER), nl(HELPER_CODE) + newline + nl(ANCHOR_HELPER))
    text = text.replace(nl(ANCHOR_FORK), nl(FORK_CODE))
    text = text.replace(nl(ANCHOR_CREATE), nl(CREATE_CODE))

    assert_no_dbsc_lead_at_eol(text, path)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)

    print("[nuitka-patch] 已给 %s 打上 MSIX/APPX 物理路径补丁（Nuitka %s）" % (path, version))
    print("[nuitka-patch] 真源备份：%s；回滚方式：把备份复制回引导层源码即可。" % backup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
