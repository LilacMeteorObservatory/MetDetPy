"""管理和文件IO相关的方法，如图像读写，路径和后缀名的常用方法等。
"""

import os
from os.path import join as path_join
from os.path import split as path_split
from typing import Optional, Union

import cv2
import numpy as np
import rawpy
from numpy.typing import NDArray

from .imgproc import Transform, contrast_stretch_uint16, contrast_stretch_uint8, scale2tgt_mean
from .metlog import BaseMetLog, get_useable_logger
from .utils import WORK_PATH, U8Mat, transpose_wh

COLOR_PATH_MAPPING = {"sRGB": os.path.join(WORK_PATH, "resource", "sRGB.icc")}
SUPPORT_COMMON_FORMAT = ["jpg", "png", "jpeg", "tiff", "tif", "bmp"]
SUPPORT_RAW_FORMAT = ["cr2", "cr3", "nef", "arw", "rw2", "raf", "dng"]
SUPPORT_ALL_IMG_FORMAT = SUPPORT_COMMON_FORMAT + SUPPORT_RAW_FORMAT


def is_ext_with(path: str, ext: str):
    """判断给定路径/文件是否以指定后缀名结尾。大小写不敏感。
    """
    return path.lower().endswith(ext.lower())


def is_ext_within(path: str, ext_list: list[str]):
    """判断给定路径/文件是否以指定后缀名列表中的某个后缀名结尾。大小写不敏感。
    """
    return path.split(".")[-1].lower() in [e.lower() for e in ext_list]


def replace_path_ext(src_path: str, ext: str):
    """替换给定路径/文件的后缀名。
    """
    return os.path.splitext(src_path)[0] + "." + ext


def change_file_path(src_path: str, tgt_path: str):
    """更改src_path下的文件到tgt_path路径下。
    
    Example: 
    
    ```
    change_file_path("/path/from/src/1.jpg","/path/tgt")
    ```
    ```
    >>> "/path/tgt/1.jpg"
    ```
    """
    return path_join(tgt_path, path_split(src_path)[-1])


def save_path_handler(save_path: str, filename: str, ext: str = "json") -> str:
    """处理保存路径。遵循以下逻辑：
    如果 save_path 是文件夹，则将 filename 更改后缀后连接到给定路径下。
    如果 save_path 指向一个存在的路径+新的文件，则直接保存。
    
    Args:
        save_path (str): 保存路径
        filename (str): 文件名
        ext (str, optional): 文件后缀名. Defaults to "json".

    Returns:
        str: 完整路径
    """
    if filename.endswith("/") or filename.endswith("\\"):
        filename = filename[:-1]
    # 若路径为文件夹，则在文件夹下保存文件
    if os.path.isdir(save_path):
        return change_file_path(replace_path_ext(filename, ext), save_path)

    # 检查父路径是否存在
    root_path, filename = path_split(save_path)
    if os.path.isdir(root_path):
        return save_path
    raise ValueError(f"Invalid saving path: {save_path}.")


def save_img(img: U8Mat,
             filename: str,
             quality: int,
             compressing: int,
             color_space: Optional[str] = None,
             logger: Optional[BaseMetLog] = None):
    logger = get_useable_logger(logger)
    if is_ext_with(filename, "png"):
        ext = ".png"
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), compressing]
    elif is_ext_with(filename, "jpg") or is_ext_with(filename, "jpeg"):
        ext = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    else:
        raise NameError(f"Unsupported suffix \"{filename.split('.')[-1]}\";"
                        "Only .png and .jpeg/.jpg are supported.")
    status, buf = cv2.imencode(ext, img, params)
    if not status:
        raise Exception("imencode failed.")
    if color_space in COLOR_PATH_MAPPING:
        try:
            import pyexiv2
            color_profile_path = COLOR_PATH_MAPPING[color_space]
            colorprofile = b""
            logger.debug(f"Load color space from: {color_profile_path}")
            if os.path.isfile(color_profile_path):
                with open(color_profile_path, mode='rb') as f:
                    colorprofile = f.read()
                with pyexiv2.ImageData(buf.tobytes()) as image_data:
                    image_data.modify_icc(colorprofile)
                    with open(filename, mode='wb') as f:
                        f.write(image_data.get_bytes())
                    return
            else:
                logger.warning(
                    f"Failed to load {color_space} config. Save without color space..."
                )

        except (ImportError, OSError):
            logger.warning(
                "Failed to load pyexiv2. EXIF data and colorprofile can not be written to files."
            )
        except Exception as e:
            logger.error(f"Fatal error: {e.__repr__()}.")
    # 降级直接写入文件
    with open(filename, mode='wb') as f:
        f.write(buf.tobytes())


def load_8bit_image(filename: str):
    img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8),
                       cv2.IMREAD_UNCHANGED)
    if img is None:
        raise Exception(f"Failed to load image: {filename}.")
    return img


def load_raw_image(filename: str,
                   auto_gamma: bool = False,
                   auto_wb: bool = False) -> NDArray[np.uint16]:
    """load cammon raw camera file with rawpy, returning an Uint16 numpy array.

    Args:
        filename (str): path to the input file.
        auto_gamma (bool, optional): whether to apply gamma transformation. Defaults to False.

    Raises:
        Exception: path not found.

    Returns:
        NDArray[np.uint16]: loaded image in uint16 format.
    """
    if not os.path.isfile(filename):
        raise Exception(f"Raw file not found: {filename}.")
    with rawpy.imread(filename) as raw:
        if auto_gamma:
            img = raw.postprocess(output_bps=16,
                                  use_camera_wb=not auto_wb,
                                  use_auto_wb=auto_wb,
                                  no_auto_bright=True,
                                  output_color=rawpy.rawpy.ColorSpace(4))
        else:
            # produce linear 16-bit RGB then run existing scale->nonlinear mapping
            img = raw.postprocess(output_bps=16,
                                  gamma=(1.0, 0.0),
                                  use_camera_wb=not auto_wb,
                                  use_auto_wb=auto_wb,
                                  no_auto_bright=True)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def load_raw_with_preprocess(
        filename: str,
        power: float = 2.222,
        target_nl_mean: float = 0.3,
        contrast_alpha: float = 1.2,
        output_bps: int = 16) -> Union[U8Mat, NDArray[np.uint16]]:
    """Load raw image and preprocess it to a non-linear image.

    Args:
        filename (str): path to the input file.
        power (float, optional): _description_. Defaults to 2.222.
        target_nl_mean (float, optional): _description_. Defaults to 0.3.
        output_bps (int, optional): _description_. Defaults to 16.

    Returns:
        NDArray[np.uint16]: _description_
    """
    img_linear = load_raw_image(filename, auto_gamma=False, auto_wb=True)
    img = scale2tgt_mean(img_linear,
                         power=power,
                         target_nl_mean=target_nl_mean)
    if output_bps == 8:
        img = (img // 257).astype(np.uint8)
        img = contrast_stretch_uint8(img, alpha=contrast_alpha)
    elif output_bps == 16:
        img = contrast_stretch_uint16(img, alpha=contrast_alpha)
    return img


def load_mask(mask_fname: Optional[str] = None,
              opencv_resize: Optional[list[int]] = None,
              grayscale: bool = False) -> U8Mat:
    """
    Load mask from the given path `mask_fname` and rescale it to the given size (if required).
    If `None` is provided, then a all-one mask will be returned.
        
    Args:
        mask_fname (Optional[str], optional): path to the mask. Defaults to None.
        opencv_resize (Optional[list[int]], optional): required resize params (in opencv style, W x H). Defaults to None.
        grayscale (bool, optional): whether to return a grayscale mask (1-channel ndarray). Defaults to False.

    Raises:
        ValueError: raised when mask_fname and opencv_resize are both empty.

    Returns:
        np.ndarray: the resized mask.
    """

    if mask_fname == None:
        if opencv_resize is None:
            raise ValueError(
                "opencv_resize is required when mask_fname is empty!")
        if grayscale:
            return np.ones(transpose_wh(opencv_resize), dtype=np.uint8)
        else:
            return np.ones(transpose_wh(opencv_resize + [3]), dtype=np.uint8)
    mask = load_8bit_image(mask_fname)
    mask_transformer = Transform()
    if opencv_resize:
        mask_transformer.opencv_resize(opencv_resize)
    if is_ext_with(mask_fname, ".jpg"):
        mask_transformer.opencv_BGR2GRAY()
        mask_transformer.opencv_binary(128, 1)
    elif is_ext_with(mask_fname, ".png"):
        # 对于png，仅取透明度层，且逻辑取反
        mask = mask[:, :, -1]
        mask_transformer.opencv_binary(128, 1, inv=True)

    if not grayscale:
        mask_transformer.expand_3rd_channel(3)

    return mask_transformer.exec_transform(mask)
