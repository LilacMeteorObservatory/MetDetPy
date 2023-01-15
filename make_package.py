# 简易的打包工具
# 用于将本项目封装为一个（数个）可执行文件。
# 支持nuitka和pyinstaller两种打包方式。新版本的pyinstaller与nuitka

import subprocess
import argparse
import time
import sys
import shutil
import zipfile
import pathlib


def run_cmd(command):
    t_start = time.time()
    ret = subprocess.run(command)
    t_end = time.time()
    return ret.returncode, t_end - t_start


def file_to_zip(path_original, z):
    '''
    Copied from https://blog.51cto.com/lanzao/4994053
     作用：压缩文件到指定压缩包里
     参数一：压缩文件的位置
     参数二：压缩后的压缩包
    '''
    f_list = list(pathlib.Path(path_original).glob("**/*"))
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
                       type=str,
                       default="pyinstaller")
argparser.add_argument(
    "--mingw64",
    help=
    "Use mingw64 as compiler. This option only works for nuitka under Windows.",
    default=False)
argparser.add_argument(
    "--onefile",
    help="Package python codes to one file. Not working currently.",
    default=False)
argparser.add_argument("--version",
                       type=str,
                       help="Software version.",
                       default="1.3.1")

args = argparser.parse_args()
compile_tool = args.tool
version = args.version

# 根据平台决定确定编译/打包后的程序后缀
platform = platform_mapping[sys.platform]
exec_suffix = ""
if (platform == "win"):
    exec_suffix = ".exe"


t0 = time.time()

if compile_tool == "nuitka":

    print("Use nuitka as package tools.")

    compile_tool = ["python", "-m", "nuitka"]
    # 构建nuitka的选项列表
    # nuitka编译的结果产生在dist/[filename].dist路径下
    nuitka_cfg = {
        "--standalone": True,
        "--show-memory": True,
        "--show-progress": True,
        "--nofollow-imports": True,
        "--remove-output": True,
        "--follow-import-to": "MetLib",
        "--output-dir": "dist",
    }

    # 根据编译平台选择是否启用mingw64
    if (platform == "win") and args.tool == "nuitka" and args.mingw64:
        print("Apply mingw64 as compiler.")
        nuitka_cfg["--mingw64"] = True

    # 编译主要检测器core.py
    target = ["core.py"]

    options = [
        key if value == True else f'{key}={value}'
        for key, value in nuitka_cfg.items() if value
    ]

    merged = compile_tool + options + target
    ret_code, time_cost = run_cmd(merged)

    print(
        f"Compiled finished with return code = {ret_code}. Time cost = {time_cost:.2f}s."
    )

    ## postprocessing
    # rename executable file and folder
    print("Renaming dist files...", end="")
    shutil.move(f"./dist/core.dist/core{exec_suffix}",
                f"./dist/core.dist/MetDetPy{exec_suffix}")
    shutil.move("./dist/core.dist", "./dist/MetDetPy")
    print("Done.")
    # copy configuration file
    print("Copy config json file...", end="")
    shutil.copy("./config.json", "./dist/MetDetPy/")
    print("Done.")
    # package codes with zip(by default).
    zip_fname = f"dist/MetDetPy_{platform}_{version}.zip"
    print(f"Zipping files to {zip_fname} ...", end="")
    with zipfile.ZipFile(zip_fname, mode='w') as zipfile_op:
        file_to_zip("dist/MetDetPy", zipfile_op)
    print("Done.")

else:
    # 使用pyinstaller作为打包工具
    print("Use nuitka as package tools.")
    compile_tool = ['pyinstaller']

    # 使用主要配置文件core.spec 打包主要检测器core.py
    # pyinstaller打包后创建文件于dist/MetDetPy目录下
    
    target = ['core.spec']
    ret_code, time_cost = run_cmd(compile_tool + target)

    print(
        f"Package finished with return code = {ret_code}. Time cost = {time_cost:.2f}s."
    )
    
    ## postprocessing
    # remove build folder
    print("Removing build files...", end="")
    shutil.rmtree(f"./build")
    print("Done.")
    # copy configuration file
    print("Copy config json file...", end="")
    shutil.copy("./config.json", "./dist/MetDetPy/")
    print("Done.")
    # package codes with zip(by default).
    zip_fname = f"dist/MetDetPy_{platform}_{version}.zip"
    print(f"Zipping files to {zip_fname} ...", end="")
    with zipfile.ZipFile(zip_fname, mode='w') as zipfile_op:
        file_to_zip("dist/MetDetPy", zipfile_op)
    print("Done.")

print(f"Package script finished. Total time cost {(time.time()-t0):.2f}s.")