# Nuitka onefile MSIX/APPX 引导层补丁

给 Nuitka 的 Windows onefile 引导层打补丁，修掉 **Windows 商店版（APPX / MSIX）**
里引擎启动即退（`0xC0000135`，`STATUS_DLL_NOT_FOUND`）的问题。

补丁本体与 MeteorMasterDetecterAI 2.5 用的是同一条思路（见
`U:\yolo\MeteorMasterDetecterAI\nuitka-patches`），针对 MetDetPy 的构建环境
（Nuitka 4.0.5）做了版本锁定与标记适配。

## 故障现象

商店版里引擎一起来就退，日志里没有 Python traceback：

| 退出码 | 含义 |
|---|---|
| `0xC0000135`（3221225781） | `STATUS_DLL_NOT_FOUND` —— 本次定位到的根因 |
| `0xC0000409`（3221226505） | fail-fast 原生异常，若修复后仍出现请单独取证 |

## 根因

Windows 商店版（MSIX / APPX）带**包文件虚拟化**：打包应用写 `%LOCALAPPDATA%`
下的路径时，实际落在包私有目录。

```
逻辑路径 A = %LOCALAPPDATA%\MetDetPy\<VERSION>
物理路径 B = %LOCALAPPDATA%\Packages\<PFN>\LocalCache\Local\MetDetPy\<VERSION>
```

（缓存目录默认以 `VERSION` 为标识，同版本多次构建需要隔离时用 `--build-tag` 覆盖。）

onefile 引导层把载荷解压到 A（物理上落在 B），随后用**逻辑路径**启动内部子进程。
Windows 加载器把逻辑路径 A 当成「应用目录」去查依赖 DLL（`python310.dll`），
该文件只存在于 B → 子进程在 Python 解释器加载前就以 `0xC0000135` 退出。

## 修法

启动内部子进程前，把逻辑路径解析成最终物理路径，用物理路径作为
`CreateProcessW` 的 `lpApplicationName`：

```
CreateFileW(逻辑路径, 0, FILE_SHARE_READ|WRITE|DELETE, OPEN_EXISTING)   // 仅查询属性
  → GetFinalPathNameByHandleW(FILE_NAME_NORMALIZED | VOLUME_NAME_DOS)
  → 去掉 \\?\ / \\?\UNC\ 前缀
  → CreateProcessW(物理路径, ...)
```

这样子进程的「应用目录」指向真实目录 B，`python310.dll` 就在旁边。
解析失败时回退到原逻辑路径，非商店环境行为不变。

## 用法

`MetLib/packaging/nuitka.py` 在 **Windows + onefile** 构建前会自动调用本脚本；
构建完成后还会检查每个产物里是否存在 `BINARY_MARKER`
（`METDETPY_MSIX_ONEFILE_PATCH_ACTIVE`），缺失直接判构建失败。

也可以单独执行（幂等）：

```bash
python nuitka-patches/patch_onefile_bootstrap.py
```

## 自我保护

1. **锁版本**：只接受 Nuitka `4.0.5`。版本不符直接失败（不猜）。
2. **锁内容**：三处锚点必须各命中且仅命中一次，否则直接失败。
3. **锁编码**：注入的 C 代码必须是纯 ASCII（MSVC 在 936 代码页下会把
   UTF-8 尾字节当 DBCS 前导字节，吞掉换行 → `unterminated #endif`）。
4. **幂等 + 确定性重放**：始终以 `.metdetpy-orig` 备份为真源重新打补丁。
5. **可回滚**：把 `OnefileBootstrap.c.metdetpy-orig` 复制回去即可还原。

## 验收要点

补丁只是「必要条件」，是否彻底解决必须由 **APPX 无缓存首启** 验收：

- 新账号 / 隔离测试机 + 新版本（或显式指定新的 `--build-tag`），普通缓存与包私有缓存**都不存在**
- 直接在软件里首次识别成功，**不预跑、不提权、不手工补 DLL**
- PML 里能看到 `python310.dll` 的有效 Load Image，且实际路径与期望缓存一致
- 回归：直播、视频分析、主动停止、模型切换、CPU / DirectML、中文与空格路径、
  升级与旧缓存共存

非商店环境下，引导层会往 stderr 打一行
`[MetDetPy onefile] METDETPY_MSIX_ONEFILE_PATCH_ACTIVE fork target: logical="..." final="..."`，
可用于确认补丁生效（`logical` 与 `final` 都指向 `%LOCALAPPDATA%\MetDetPy\<VERSION>`）。
