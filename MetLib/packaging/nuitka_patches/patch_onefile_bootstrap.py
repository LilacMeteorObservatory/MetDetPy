"""Isolate and patch Nuitka's Windows onefile bootstrap for one build."""

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
import sys

# Versions previously used for project builds. Other versions warn but must
# still pass every source anchor and encoding check.
SUPPORTED_NUITKA_VERSIONS = frozenset({"4.0.5"})

SOURCE_MARKER = "METDETPY_MSIX_ONEFILE_PATCH"

# 写进产物二进制的标记字符串，构建后由 nuitka.py 检查，防止静默漏打。
# The packaging backend imports this constant for artifact verification.
BINARY_MARKER = SOURCE_MARKER + "_ACTIVE"

# 运行时诊断开关：设置该环境变量（值为 1/yes/true 等）才打印 fork 目标路径。
# Fork diagnostics are opt-in. They used to print on every launch, and launchers
# that surface any child stderr as a warning showed a bogus warning every run.
VERBOSE_ENV_VAR = "METDETPY_ONEFILE_VERBOSE"

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
    // Fork diagnostics are opt-in. They used to be printed on every launch,
    // which made launchers that treat any child stderr as a warning show a
    // bogus warning for every run. Set METDETPY_ONEFILE_VERBOSE=1 (or any
    // value starting with 1/y/t) to get the line back when troubleshooting.
    // stdout carries the JSON line protocol with the detection results and
    // stays clean either way.
    {
        wchar_t verbose_value[8] = {0};
        DWORD const verbose_length =
            GetEnvironmentVariableW(L"METDETPY_ONEFILE_VERBOSE", verbose_value, 8);
        if (verbose_length > 0 && verbose_length < 8) {
            wchar_t const first = verbose_value[0];
            if (first == L'1' || first == L'y' || first == L'Y' || first == L't' || first == L'T') {
                fprintf(stderr,
                        "[MetDetPy onefile] METDETPY_MSIX_ONEFILE_PATCH_ACTIVE fork target: logical=\"%ls\" final=\"%ls\"\n",
                        (wchar_t const *)fork_binary, (wchar_t const *)fork_binary_final);
                fflush(stderr);
            }
        }
    }

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


def patch_bootstrap(path: Path, version: str) -> None:
    """Patch an explicit private copy; never discover or edit site-packages."""
    if version not in SUPPORTED_NUITKA_VERSIONS:
        print(
            f"[nuitka-patch] WARNING: Nuitka {version} is outside the version "
            "allowlist; continuing with strict source anchor checks.",
            file=sys.stderr,
        )

    assert_ascii(HELPER_CODE, "HELPER_CODE")
    assert_ascii(FORK_CODE, "FORK_CODE")
    assert_ascii(CREATE_CODE, "CREATE_CODE")

    if BINARY_MARKER not in CREATE_CODE:
        fail("CREATE_CODE 里缺少 BINARY_MARKER，构建后的二进制校验会失效。")
    if VERBOSE_ENV_VAR not in CREATE_CODE:
        fail("CREATE_CODE 里缺少 VERBOSE_ENV_VAR，运行时诊断开关会失效。")

    with open(path, "r", encoding="utf-8", newline="") as handle:
        text = handle.read()
    source = path
    if SOURCE_MARKER in text:
        fail(f"{path} already contains the patch. Restore a clean Nuitka installation first.")

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

    print(f"[nuitka-patch] Patched private bootstrap: {path} (Nuitka {version})")


@contextmanager
def isolated_nuitka(env=None):
    """Yield a child environment selecting a patched, temporary Nuitka copy.

    Preserve existing PYTHONPATH entries, including pyexiv2 staging. Clean up
    on copy, patch, probe and compile failures as well as successful builds.
    """
    import importlib.util

    spec = importlib.util.find_spec("nuitka")
    if spec is None or not spec.submodule_search_locations:
        fail(f"Nuitka is not installed for {sys.executable}")
    source = Path(next(iter(spec.submodule_search_locations))).resolve()
    version = get_nuitka_version()
    with tempfile.TemporaryDirectory(prefix="metdetpy-nuitka-") as stage:
        private_package = Path(stage) / "nuitka"
        shutil.copytree(source, private_package,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.metdetpy-orig"))
        patch_bootstrap(private_package / "build" / "static_src" / "OnefileBootstrap.c", version)
        child_env = dict(env or {})
        previous_path = child_env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
        child_env["PYTHONPATH"] = stage + (os.pathsep + previous_path if previous_path else "")
        probe_env = os.environ.copy()
        probe_env.update(child_env)
        subprocess.run([
            sys.executable, "-c",
            "import sys, nuitka; from pathlib import Path; "
            "actual = Path(nuitka.__file__).resolve().parent; "
            "expected = Path(sys.argv[1]).resolve(); "
            "assert actual == expected, (actual, expected); "
            "print('Using isolated Nuitka:', actual)",
            str(private_package),
        ], env=probe_env, check=True)
        yield child_env
