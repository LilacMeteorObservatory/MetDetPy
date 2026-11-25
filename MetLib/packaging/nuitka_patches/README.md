# Nuitka onefile MSIX/APPX 引导层补丁

Windows + onefile 构建自动使用此补丁。它在启动内部子进程前，通过
`CreateFileW` 和 `GetFinalPathNameByHandleW` 解析可执行文件的实际物理路径，
再传给 `CreateProcessW`，用于规避 MSIX/APPX 路径虚拟化下的 DLL 查找问题
（`0xC0000135` / `STATUS_DLL_NOT_FOUND`）。解析失败时保留原路径。

## 构建隔离

`MetLib/packaging/nuitka.py` 调用 `isolated_nuitka()`：

1. 将当前解释器安装的整个 `nuitka` 包复制到临时目录。
2. 只修补副本中的 `build/static_src/OnefileBootstrap.c`。
3. 为编译子进程设置 `PYTHONPATH`，优先选择副本，同时保留 pyexiv2
   暂存目录和已有路径；不修改父进程环境。
4. 通过独立 Python 子进程核对实际导入的 `nuitka.__file__`。
5. 三个入口共用本次构建的副本，成功或异常退出时均清理临时目录。

安装目录保持只读，不创建或重用 `.metdetpy-orig`。如果已安装的源码带有
旧补丁标记，构建失败；应先恢复或重新安装干净的 Nuitka，不自动信任旧备份。
三个程序共享的运行时解压缓存保持不变，这与编译器副本的隔离相互独立。

入口仍为：

```powershell
python make_package.py --backend nuitka --onefile --apply-zip --build-report
```

补丁模块是打包内部实现，不再提供修改已安装 Nuitka 的独立命令。

## 运行时诊断

补丁默认静默，正式环境和商店版都不会向 stderr 写任何内容：启动器（如
MeteorMasterAI）会把子进程的任何 stderr 当作警告展示，无条件打印会导致每次
启动都出现一条误报。排查时临时打开开关即可：

```powershell
$env:METDETPY_ONEFILE_VERBOSE = "1"
```

打开后引导层会在启动子进程前向 stderr 打印一行

```
[MetDetPy onefile] METDETPY_MSIX_ONEFILE_PATCH_ACTIVE fork target: logical="..." final="..."
```

非商店环境下 `logical` 与 `final` 相同（`%LOCALAPPDATA%\MetDetPy\<BUILD_TAG>`）；
商店环境下 `final` 指向 `%LOCALAPPDATA%\Packages\<PFN>\LocalCache\Local\MetDetPy\<BUILD_TAG>`，
可据此确认路径虚拟化确实发生。诊断只写 stderr，stdout 始终只保留 JSON 行协议。

## 兼容性检查

- `SUPPORTED_NUITKA_VERSIONS` 是版本白名单，初始保留历史构建版本 `4.0.5`。
- 白名单外版本打印 warning 后继续，**不因版本号直接阻断**；这不代表已经验证兼容。
- 无论版本是否在白名单，三个源码锚点均必须各匹配一次，否则失败。
- 注入的 C 代码必须是 ASCII，并保留源码编码检查。
- 编译后逐个检查三个 exe 中的 `METDETPY_MSIX_ONEFILE_PATCH_ACTIVE` 标记。
- 使用 UPX 时，先搜索原文件；未命中则在输出目录所属磁盘上逐个临时解压后检查，
  不修改发布 exe，检查后删除临时文件。文件缺失、解压失败和标记缺失分别报告。

标记只能证明补丁进入产物。商店版还需在新账号或隔离机上完成无缓存首启验收：
普通缓存和包私有缓存均不存在，不预跑、不提权、不手工补 DLL，确认实际 DLL
加载路径，并回归中文、空格路径以及 CPU / DirectML 检测流程。
