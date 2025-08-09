from typing import Any, Optional, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from .metlog import BaseMetLog, get_useable_logger
from .metstruct import DenoiseOption
from .utils import EULER_CONSTANT, FastGaussianParam, U8Mat, circular_kernel
from .videoloader import BaseVideoLoader, VanillaVideoLoader

class BaseImgContainer(object):
    """ImgContainer is a class that receives stream inputs (by its `append`) 
    method and provides the result by `export` method.
    
    It can be used to maintain a stream statistics data,
    or simply store all inputs in a list.
    """

    def __init__(self):
        self.container = None

    def append(self, new_frame: U8Mat):
        pass

    def export(self):
        return self.container


class AllImgContainer(BaseImgContainer):

    def __init__(self):
        self.container: list[U8Mat] = list()

    def append(self, new_frame: U8Mat):
        self.container.append(new_frame)


class MaxImgContainer(BaseImgContainer):

    def append(self, new_frame: U8Mat):
        if self.container is None:
            self.container = new_frame
        else:
            self.container = np.max([self.container, new_frame], axis=0)


class FastGaussianContainer(BaseImgContainer):

    def append(self, new_frame: U8Mat):
        fg_frame = FastGaussianParam(new_frame.astype(np.uint16))
        if self.container is None:  # type: ignore
            self.container = fg_frame
        else:
            self.container += fg_frame


def single_sigma_clipping(img_list: list[U8Mat],
                          ref_fg_img: FastGaussianParam,
                          sigma_high: float = 3.0,
                          sigma_low: float = 3.0) -> FastGaussianParam:
    mu, std = ref_fg_img.mu, np.sqrt(ref_fg_img.var)
    rej_high_thre = np.round(mu + sigma_high * std).clip(0,
                                                         255).astype(np.uint8)
    rej_low_thre = np.round(mu - sigma_low * std).clip(0, 255).astype(np.uint8)

    fgp_clipped = None
    for img in img_list:
        mask = (img > rej_high_thre) | (img < rej_low_thre)
        fgp_img = FastGaussianParam(img.astype(np.uint16))
        fgp_img.mask(mask)
        if fgp_clipped is None:
            fgp_clipped = fgp_img
        else:
            fgp_clipped += fgp_img
    if fgp_clipped is None:
        return ref_fg_img
    return (ref_fg_img - fgp_clipped)


def get_gumbel_mean(n: int) -> float:
    """return the mean value for n-sample gumbel distribution.

    Args:
        n (int): number of samples.

    Returns:
        float: the mean value for n-sample gumbel distribution.
    """
    sqrt2logn: float = np.sqrt(2 * np.log(n))
    return (sqrt2logn - (np.log(np.log(n)) + np.log(4 * np.pi)) /
            (2 * sqrt2logn) + EULER_CONSTANT / sqrt2logn)


def fill_large_contours(src: U8Mat, max_allow_area: int = 30):
    """
    填充二值图像中面积大于指定阈值的封闭区域
    
    参数:
        input_image_path: 输入二值图像路径
        output_image_path: 输出图像路径
        min_area: 最小面积阈值，默认为100像素
    """
    contours, hierarchy = cv2.findContours(src, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > max_allow_area and hierarchy[0][i][
                3] != -1:  # hierarchy[0][i][3] != -1 表示是内轮廓(封闭区域)
            cv2.drawContours(src, [cnt], 0, [0, 0, 0], -1)

    return src


def _batch_stacker(video_loader: BaseVideoLoader,
                   ImgContainerClsList: list[type[BaseImgContainer]],
                   start_frame: Optional[int] = None,
                   end_frame: Optional[int] = None,
                   input_logger: Optional[BaseMetLog] = None) -> list[Any]:
    logger = get_useable_logger(input_logger)
    container_list = [x() for x in ImgContainerClsList]
    try:
        if start_frame != None or end_frame != None:
            video_loader.reset(start_frame=start_frame, end_frame=end_frame)
        base_shape = None
        video_loader.start()
        for _ in range(video_loader.iterations):
            img_frame = video_loader.pop()
            if img_frame is None: break
            if base_shape is None:
                base_shape = img_frame.shape
            elif base_shape != img_frame.shape:
                raise ValueError(
                    f"Expect new frame has the same shape " +
                    f"as the base frame {base_shape}, got {img_frame.shape}.")
            for container in container_list:
                container.append(img_frame)
    except Exception as e:
        logger.error(e.__repr__())
        return [x.container for x in container_list]
    finally:
        video_loader.stop()

    return [x.container for x in container_list]


def all_stacker(video_loader: BaseVideoLoader,
                start_frame: Optional[int] = None,
                end_frame: Optional[int] = None,
                logger: Optional[BaseMetLog] = None) -> Optional[list[U8Mat]]:
    """ Load all frames to a matrix(list, actually).

    Args:
        video_loader (BaseVideoLoader): initialized video loader.
        start_frame (Optional[int], optional): the start frame of the stacker. Defaults to None.
        end_frame (Optional[int], optional): the end frame of the stacker.. Defaults to None.
        logger (Optional[BaseMetLog], optional): a logging.Logger object for logging use. Defaults to None.

    Returns:
        list: a list containing all frames.
    """
    return _batch_stacker(video_loader, [AllImgContainer], start_frame,
                          end_frame, logger)[0]


def max_stacker(video_loader: VanillaVideoLoader,
                start_frame: Optional[int] = None,
                end_frame: Optional[int] = None,
                logger: Optional[BaseMetLog] = None) -> Optional[U8Mat]:
    """Stack frames within range and return a stacked image.

    Args:
        video_loader (BaseVideoLoader): initialized video loader.
        start_frame (Optional[int], optional): the start frame of the stacker. Defaults to None.
        end_frame (Optional[int], optional): the end frame of the stacker.. Defaults to None.
        logger (Optional[BaseMetLog], optional): a logging.Logger object for logging use. Defaults to None.

    Returns:
        Optional[NDArray]: the stacked image. If there is no frames to stack, return None.
    """
    return _batch_stacker(video_loader, [MaxImgContainer], start_frame,
                          end_frame, logger)[0]


def dust_and_scratches(img: U8Mat, radius: int, threshold: int):
    """
    模拟 Photoshop '蒙尘与划痕' 滤镜
    img: BGR 图像
    radius: 中值滤波半径
    threshold: 亮度差阈值
    coded by GPT-4o
    """
    # to Lab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(lab)
    # 中值滤波
    median_L = cv2.medianBlur(L, 2 * radius + 1)
    # 中值替换
    diff_L = cv2.absdiff(L, median_L)
    mask_keep = diff_L > threshold
    L_result = L.copy()
    L_result[mask_keep] = median_L[mask_keep]
    # back2 RGB
    lab_result = cv2.merge([L_result, A, B])
    result = cv2.cvtColor(lab_result, cv2.COLOR_Lab2BGR)

    return result


def connect_highlight_area(light_img: U8Mat,
                           light_diff_img: Union[NDArray[np.float64], U8Mat],
                           rep_times: int = 1,
                           kernel_size: int = 31,
                           clip_threshold: int = 30,
                           logger: Optional[BaseMetLog] = None) -> U8Mat:
    """尝试连接图像中的断线。

    Args:
        light_img (U8Mat): 图像亮场。
        light_diff_img (NDArray[np.float64]): 图像亮部差值图像。
        rep_times (int, optional): 形态学运算循环次数. Defaults to 1.
        kernel_size (int, optional): 使用的形态学核尺寸. Defaults to 31.
        highlight_multipier (float, optional): _description_. Defaults to 1.2.
        logger (Optional[BaseMetLog], optional): Logger. Defaults to None.

    Returns:
        U8Mat: 连接了断线的图像亮场。
    """
    logger = get_useable_logger(logger)
    # Otsu法用于分离显著明亮区域。但如果只是常规叠加，可能不适合。
    clipped_diff_img = cv2.cvtColor(
        np.clip(light_diff_img, clip_threshold, 255).astype(np.uint8),
        cv2.COLOR_BGR2GRAY)
    otsu_thresh, binary_highlight_mask = cv2.threshold(
        clipped_diff_img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    logger.debug(f"Extrame highlight threshold = {otsu_thresh:.2f}")
    masked_light_img = light_img * binary_highlight_mask[..., None]
    init_binary_mask = np.copy(binary_highlight_mask)
    # 对Mask和亮场执行闭运算
    close_kernel = circular_kernel(kernel_size)
    binary_highlight_mask = cv2.morphologyEx(binary_highlight_mask,
                                             cv2.MORPH_CLOSE,
                                             close_kernel,
                                             iterations=rep_times)
    masked_light_img = cv2.morphologyEx(masked_light_img,
                                        cv2.MORPH_CLOSE,
                                        close_kernel,
                                        iterations=rep_times)
    # Mask亮场并叠加
    masked_hat = binary_highlight_mask - init_binary_mask
    # mask搜索并移除面积较大的区域（通常可能是错误填充）
    masked_hat = fill_large_contours(masked_hat, 20)
    ext_light_img = masked_light_img * masked_hat[..., None]
    return np.max([light_img, ext_light_img], axis=0)


def mfnr_mix_stacker(video_loader: VanillaVideoLoader,
                     denoise_cfg: DenoiseOption,
                     start_frame: Optional[int] = None,
                     end_frame: Optional[int] = None,
                     logger: Optional[BaseMetLog] = None) -> Optional[U8Mat]:
    """混合优化叠加算法，基于多帧统计值混合平均值和最大值叠加图像，支持（实验性）连接断线等。

    Args:
        video_loader (VanillaVideoLoader): _description_
        start_frame (Optional[int], optional): _description_. Defaults to None.
        end_frame (Optional[int], optional): _description_. Defaults to None.
        logger (Optional[BaseMetLog], optional): _description_. Defaults to None.

    Returns:
        Optional[np.ndarray]: _description_
    """
    logger = get_useable_logger(logger)
    highlight_preserve, blur_ksize = denoise_cfg.highlight_preserve, denoise_cfg.blur_ksize
    connect_cfg, mfnr_param = denoise_cfg.connect_lines, denoise_cfg.mfnr_param
    logger.debug("Load image stack from files...")
    max_img, img_stack, init_fg_img = _batch_stacker(
        video_loader,
        [MaxImgContainer, AllImgContainer, FastGaussianContainer], start_frame,
        end_frame, logger)
    logger.debug("Apply single sigma-clipping...")
    sc_avg_img = single_sigma_clipping(img_stack,
                                       init_fg_img,
                                       sigma_high=3.0,
                                       sigma_low=3.0)
    logger.debug("Calculate gumbel-dist parameters...")

    gumble_mean = get_gumbel_mean(len(img_stack))
    # 计算实际最大亮度与预期最大亮度的差异 => max_bias_diff_img
    expect_max_upper: NDArray[np.float64] = (
        sc_avg_img.mu + np.mean(np.sqrt(sc_avg_img.var)) * gumble_mean *
        mfnr_param.bg_fix_factor)
    max_bias_diff_img: NDArray[np.float64] = max_img.astype(
        np.float64) - expect_max_upper
    # 计算有高离群的亮均值作为阈值，和高光部分求并集作为前景组分
    highlight_avg_diff: np.float64 = np.average(
        max_bias_diff_img[max_bias_diff_img > 0])
    highlight_area: NDArray[np.bool_] = (max_img > 255 * highlight_preserve)
    fg_mask = (max_bias_diff_img > highlight_avg_diff) | highlight_area

    # Mask被灰度化后还原为三通道
    fg_mask = np.repeat((np.sum(fg_mask.astype(np.uint8), axis=-1) >= 1)[...,
                                                                         None],
                        3,
                        axis=-1).astype(float)

    # Gaussian Blur for stage1_diff
    stage1_diff_blur = cv2.GaussianBlur(fg_mask,
                                        ksize=(blur_ksize, blur_ksize),
                                        sigmaX=3)

    # 如果需要的话，尝试连接亮部的断线。
    if connect_cfg.switch:
        max_img = connect_highlight_area(max_img,
                                         max_bias_diff_img,
                                         rep_times=1,
                                         kernel_size=connect_cfg.ksize,
                                         clip_threshold=connect_cfg.threshold,
                                         logger=logger)

    # 亮度补正，高光保护
    # 亮度达到255时系数为0，下限时候为1。
    highlight_fix_factor: NDArray[np.float64] = 1 - (
        (max_img / 255 - highlight_preserve).clip(0, 1) /
        (1 - highlight_preserve))
    logger.debug(
        f"highlight fix factor = " +
        f"{(np.mean(np.sqrt(sc_avg_img.var)) * gumble_mean * mfnr_param.bg_fix_factor):.4f}"
    )
    fixed_max_img: NDArray[np.float64] = max_img - (
        (np.mean(np.sqrt(sc_avg_img.var)) * gumble_mean) *
        highlight_fix_factor)

    # 混合最大值图像和修正平均值图像
    mix_img_uint8 = np.round(fixed_max_img * stage1_diff_blur + sc_avg_img.mu *
                             (1 - stage1_diff_blur)).astype(np.uint8)
    return mix_img_uint8


def simple_denoise_stacker(
        video_loader: VanillaVideoLoader,
        denoise_cfg: DenoiseOption,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        logger: Optional[BaseMetLog] = None) -> Optional[U8Mat]:
    """简单叠加降噪算法，基于蒙尘划痕算法提取亮部，使用双边滤波降噪，支持（实验性）连接断线。

    Args:
        video_loader (VanillaVideoLoader): _description_
        start_frame (Optional[int], optional): _description_. Defaults to None.
        end_frame (Optional[int], optional): _description_. Defaults to None.
        logger (Optional[BaseMetLog], optional): _description_. Defaults to None.

    Returns:
        Optional[np.ndarray]: _description_
    """
    logger = get_useable_logger(logger)
    highlight_preserve, blur_ksize = denoise_cfg.highlight_preserve, denoise_cfg.blur_ksize
    connect_cfg, simple_cfg = denoise_cfg.connect_lines, denoise_cfg.simple_param
    max_img: U8Mat = _batch_stacker(video_loader, [MaxImgContainer],
                                    start_frame, end_frame, logger)[0]
    # 使用蒙尘划痕拆分出高亮区域
    est_bg_img = dust_and_scratches(max_img,
                                    radius=simple_cfg.ds_radius,
                                    threshold=simple_cfg.ds_threshold)

    # 计算实际最大亮度与预期最大亮度的差异 => max_bias_diff_img
    max_diff_img: NDArray[np.float64] = max_img.astype(np.float64) - est_bg_img
    # 计算有高离群的亮均值作为阈值，和高光部分求并集作为前景组分
    highlight_avg_diff: np.float64 = np.average(max_diff_img[max_diff_img > 0])
    highlight_area: NDArray[np.bool_] = (max_img > 255 * highlight_preserve)
    fg_mask = (max_diff_img > highlight_avg_diff) | highlight_area

    # Mask被灰度化后还原为三通道
    fg_mask = np.repeat((np.sum(fg_mask.astype(np.uint8), axis=-1) >= 1)[...,
                                                                         None],
                        3,
                        axis=-1).astype(float)
    fg_mask_blur = cv2.GaussianBlur(fg_mask,
                                    ksize=(blur_ksize, blur_ksize),
                                    sigmaX=3)
    # 如果需要的话，尝试连接亮部的断线。
    cp_max_img = np.asarray(max_img)
    if connect_cfg.switch:
        # 对于单张图像算法，max_diff_img则需要排除掉星点，以避免不必要的连接
        star_filter = circular_kernel(3)
        filtered_diff_img = cv2.morphologyEx(max_diff_img, cv2.MORPH_OPEN,
                                             star_filter)
        cp_max_img = connect_highlight_area(
            cp_max_img,
            filtered_diff_img,
            rep_times=1,
            kernel_size=connect_cfg.ksize,
            clip_threshold=connect_cfg.threshold,
            logger=logger)
    # 背景部分使用双边滤波，和前景混合。
    denoise_bg = cv2.bilateralFilter(max_img,
                                     d=simple_cfg.bi_d,
                                     sigmaColor=simple_cfg.bi_sigma_color,
                                     sigmaSpace=simple_cfg.bi_sigma_space)
    mixed_img = (fg_mask_blur * cp_max_img +
                 (1 - fg_mask_blur) * denoise_bg).astype(np.uint8)
    return mixed_img
