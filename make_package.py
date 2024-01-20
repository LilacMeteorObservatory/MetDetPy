# 简易的打包工具
# 用于将本项目封装为一个（数个）可执行文件。
# 支持nuitka和pyinstaller两种打包方式。推荐使用新版本的pyinstaller与nuitka

import argparse
import os
import platform as pf
import shutil
import subprocess
import sys
import time
import zipfile
from functools import partial
from pathlib import Path

from MetLib.utils import VERSION

# alias
join_path = os.path.join


def run_cmd(command):
    t_start = time.time()
    ret = subprocess.run(command)
    t_end = time.time()
    return ret.returncode, t_end - t_start


def nuitka_compile(header, options, target):
    """使用nuitka编译打包的API

    Args:
        header (str): 启动的指令，如python -m nuitka 或 nuitka （取决于平台）。
        option_list (dict): 编译选项列表。
        tgt (str): 待编译目标。
    """
    options_list = [
        key if value == True else f'{key}={value}'
        for key, value in options.items() if value
    ]

    merged = header + options_list + [
        target,
    ]

    ret_code, time_cost = run_cmd(merged)
    print(
        f"Compiled {target} finished with return code = {ret_code}. Time cost = {time_cost:.2f}s."
    )

    # 异常提前终止
    if ret_code != 0:
        print(
            f"Fatal compile error occured when compiling {target}. Compile terminated."
        )
        exit(-1)


def pyinstaller_compile(header="pyinstaller", spec=None):
    ret_code, time_cost = run_cmd([header, spec]")

    print(
        f"Package finished with return code = {ret_code}. Time cost = {time_cost:.2f}s."
    )

    # 异常提前终止
    if ret_code != 0:
        print(
            f"Fatal compile error occured when compiling {spec}. Compile terminated."
        )
        exit(-1)


def file_to_zip(path_original, z):
    '''
    Copied from https://blog.51cto.com/lanzao/4994053
     作用：压缩文件到指定压缩包里
     参数一：压缩文件的位置
     参数二：压缩后的压缩包
    '''
    f_list = list(Path(path_original).glob("**/*"))
    for f in f_list:
        z.write(f, str(f)[len(path_original):])


platform_mapping = {
    "win32": "win",
    "cygwin": "win",
    "darwin": "macos",
    "linux2": "linux",
    "linux": "linux"
}

argparser = argparse.ArgumentParser()

argparser.add_argument("--tool",
                       "-T",
                       help="Use nuitka or pyinstaller",
                       choices=['nuitka', 'pyinstaller'],
                       required=True,
                       type=str)
argparser.add_argument(
    "--mingw64",
    action="store_true",
    help="Use mingw64 as compiler. This option only works for nuitka under Windows.",
    default=False)
argparser.add_argument(
    "--apply-upx",
    action="store_true",
    help="Apply UPX to squeeze the size of executable program.",
)
argparser.add_argument(
    "--apply-zip",
    action="store_true",
    help="Generate .zip files after packaging.",
)

args = argparser.parse_args()
compile_tool = args.tool
release_version = VERSION
apply_upx = args.apply_upx
apply_zip = args.apply_zip

# 根据平台/版本决定确定编译/打包后的程序后缀

platform = platform_mapping[sys.platform]
exec_suffix = ""
if (platform == "win"):
    exec_suffix = ".exe"
if (platform == "macos"):
    mac_main_ver = int(pf.mac_ver()[0].split(".")[0])
    if mac_main_ver>=13 and compile_tool=="nuitka":
        exec_suffix = ".bin"
        platform += "13+"

# 设置工作路径，避免出现相对路径引用错误
work_path = os.path.dirname(os.path.abspath(__file__))
compile_path = join_path(work_path, "dist")

t0 = time.time()

if compile_tool == "nuitka":
    print("Use nuitka as package tools.")

    # 检查python版本 必要时启用alias python3
    compile_tool = [sys.executable, "-m", "nuitka"]
    # 将header作为偏函数打包，简化后续传参
    nuitka_compile = partial(nuitka_compile, compile_tool)

    # 构建共用的打包选项，根据编译平台选择是否启用mingw64
    nuitka_base = {
        "--no-pyi-file": True,
        "--remove-output": True,
    }
    if (platform == "win") and args.tool == "nuitka" and args.mingw64:
        print("Apply mingw64 as compiler.")
        nuitka_base["--mingw64"] = True

    # upx启用时，利用which获取upx路径
    if apply_upx:
        upx_cmd = subprocess.run(["which", "upx"])
        if upx_cmd.returncode == 0:
            nuitka_base["--plugin-enable"] = "upx"
            nuitka_base["--upx-binary"] = upx_cmd.stdout

    # 打包编译MetLib库为pyd文件
    # metlib_cfg = {
    #    "--module": True,
    #    "--output-dir": join_path(compile_path, "MetLib")
    # }
    # metlib_cfg.update(nuitka_base)
    # metlib_path = join_path(work_path, "MetLib")
    # metlib_filelist = [
    #    join_path(metlib_path, x) for x in os.listdir(metlib_path)
    #    if x.endswith(".py")
    # ]
    # for filename in metlib_filelist:
    #    if filename.endswith("__init__.py"): continue
    #    nuitka_compile(options=metlib_cfg, target=filename)

    # nuitka编译的结果产生在dist/MetDetPy.dist路径下
    met_cfg = {
        "--standalone": True,
        "--output-dir": compile_path,
    }

    met_cfg.update(nuitka_base)

    # 编译主要检测器MetDetPy.py
    nuitka_compile(met_cfg, target=join_path(work_path, "MetDetPy.py"))

    # 编译视频叠加工具ClipToolkit.py
    # 不能不编译MetLib相关文件，否则会出现非常奇怪的报错（找不到np，但直接调用MetLib所有函数都没问题）
    # 由于该问题暂时没法解决，必须全部编译。
    stack_cfg = {
        "--standalone": True,
        "--output-dir": compile_path
    }
    stack_cfg.update(nuitka_base)

    # 编译视频叠加工具ClipToolkit.py
    nuitka_compile(stack_cfg, target=join_path(work_path, "ClipToolkit.py"))

    # postprocessing
    # remove duplicate files of ClipToolkit
    print("Merging...", end="", flush=True)
    shutil.move(join_path(compile_path, "ClipToolkit.dist", f"ClipToolkit{exec_suffix}"),
                join_path(compile_path, "MetDetPy.dist"))
    shutil.rmtree(join_path(compile_path, "ClipToolkit.dist"))
    print("Done.")
    # rename executable file and folder
    # shutil.move(join_path(compile_path, "MetLib"),
    #            join_path(compile_path, "MetDetPy.dist", "MetLib"))
    print("Renaming executable files...", end="", flush=True)
    shutil.move(join_path(compile_path, "MetDetPy.dist"),
                join_path(compile_path, "MetDetPy"))
    print("Done.")

else:
    # 使用pyinstaller作为打包工具
    print("Use pyinstaller as package tools.")

    # 使用主要配置文件 MetDetPy.spec 打包主要检测器MetDetPy.py
    # pyinstaller打包后创建文件于dist/MetDetPy目录下

    pyinstaller_compile(spec='MetDetPy.spec')
    pyinstaller_compile(spec='ClipToolkit.spec')

    # postprocessing
    # remove build folder
    print("Removing build files...", end="", flush=True)
    shutil.rmtree(f"./build")
    print("Done.")
    print("Merging dist files...", end="", flush=True)
    shutil.move(join_path(compile_path, "ClipToolkit", f"ClipToolkit{exec_suffix}"),
                join_path(compile_path, "MetDetPy"))
    shutil.rmtree(join_path(compile_path, "ClipToolkit"))
    print("Done.")

# shared postprocessing
# copy configuration file
print("Copy config json folder...", end="", flush=True)
shutil.copytree("./config", "./dist/MetDetPy/config")
print("Done.")
print("Copy weights folder...", end="", flush=True)
shutil.copytree("./weights", "./dist/MetDetPy/weights")
print("Done.")
# package codes with zip(if applied).
if apply_zip:
    zip_fname = join_path(compile_path,
                          f"MetDetPy_{platform}_{release_version}.zip")
    print(f"Zipping files to {zip_fname} ...", end="", flush=True)
    with zipfile.ZipFile(zip_fname, mode='w') as zipfile_op:
        file_to_zip(join_path(compile_path, "MetDetPy"), zipfile_op)
    print("Done.")

print(f"Package script finished. Total time cost {(time.time()-t0):.2f}s.")
