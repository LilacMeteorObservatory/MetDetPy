"""
# MetVisu
管理关于可视化的绘图句柄及API。

"""
from __future__ import annotations

import dataclasses
from abc import abstractmethod
from typing import Literal, Optional, Union

import cv2
import numpy as np

from .metlog import get_default_logger
from .utils import COLOR_MAP, Transform, U8Mat, pt_offset

DEFAULT_VISUAL_DELAY = 200
DEFAULT_INTERRUPT_KEY = "q"
DEFAULT_COLOR = "white"
AS_INPUT = "as-input"

# 描述初始位置
# TODO: 稳健性需要更多测试
# in [w,h,ow,oh,dw,dh]
POSITION_MAP: dict[str, tuple[float, float, float, float, float, float]] = {
    "left": (0, 0.5, 1, 0, 0, 1),
    "left-top": (0, 0, 1, 1, 0, 1),
    "top": (0.5, 0, 0, 1, 0, 1),
    "left-bottom": (0, 1, 1, 0, 0, -1),
    "right-top": (0.8, 0, 0, 1, 0, 1),
    "right": (0.8, 0.5, 0, 0, 0, 1),
    "right-bottom": (0.8, 0.9, 0, 0, 0, -1),
    "bottom": (0.5, 0.9, 0, -1, 0, -1)
}


def parse_color(color: Union[ColorTuple, str]):
    if isinstance(color, str):
        if color in COLOR_MAP:
            color = COLOR_MAP[color]
        else:
            raise KeyError("color not found in predefined color map.")
    assert len(color) == 3, "invalid color"
    return color


def gray2colorimg(gray_image: U8Mat, color: Union[ColorTuple, str]) -> U8Mat:
    """Convert the grayscale image (h,w) to a color one (h,w,3) with the given color。

    Args:
        gray_image (U8Mat): the grayscale image.
        color (Union[ColorTuple, str]): the color tuple or a color-referring str.

    Returns:
        U8Mat: colored gray image.
    """
    color_u8 = np.array(parse_color(color), dtype=np.uint8)
    return gray_image[:, :, None] * color_u8[None, ...]


def scale_pt(pt: Union[list[int], tuple[int, int]],
             scaler: tuple[float, float]) -> list[int]:
    w_scaler, h_scaler = scaler
    return [int(pt[0] / w_scaler), int(pt[1] / h_scaler)]


########### MetVisu Dataclasses ################

LAZY_FLAG = Literal["as-input"]
ColorTuple = tuple[int, int, int]


@dataclasses.dataclass()
class BaseVisuAttrs(object):
    name: str
    sync_attributes: list[str] = dataclasses.field(default_factory=lambda: [])

    def _sync_attr(self, src: BaseVisuAttrs, attr_name: str):
        if getattr(self, attr_name, None) is not None:
            return
        src_attr = getattr(src, attr_name, None)
        if src_attr == LAZY_FLAG:
            raise ValueError(
                "as-input attribute should have a specific value.")
        setattr(self, attr_name, src_attr)

    def _batch_sync(self, src: BaseVisuAttrs, attr_name_list: list[str]):
        for attr_name in attr_name_list:
            self._sync_attr(src, attr_name)

    def sync(self, src: BaseVisuAttrs):
        """Sync all attributes from another same class object.
        If `as-input` is in the other one, then this object must 
        define a specific value. Otherwise, an error will be raised.

        Args:
            src (BaseVisuAttrs): the other one in the same class.
        """
        assert isinstance(src, self.__class__), (
            f"properties should be from {self.__class__.__name__}, " +
            f"got {src.__class__.__name__}.")
        self._batch_sync(src, self.sync_attributes)

    @abstractmethod
    def render(self, src_img: U8Mat, scaler: tuple[float, float]) -> U8Mat:
        pass

    def not_ready_error_raiser(self):
        raise ValueError(f"{self.__class__.__name__} is not ready to render!")


@dataclasses.dataclass
class ImgVisuAttrs(BaseVisuAttrs):
    weight: Optional[float] = None
    img: Optional[U8Mat] = None
    color: Union[ColorTuple, str, LAZY_FLAG, None] = None
    sync_attributes: list[str] = dataclasses.field(
        default_factory=lambda: ['weight', 'img', 'color'])

    def render(self, src_img: U8Mat, scaler: tuple[float, float]) -> U8Mat:
        if self.weight is None or self.img is None or self.color == AS_INPUT:
            self.not_ready_error_raiser()
        if len(self.img.shape) == 2:
            if self.color is not None:
                self.img = gray2colorimg(self.img, self.color)
            else:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        (src_h, src_w) = src_img.shape[:2]
        (img_h, img_w) = self.img.shape[:2]
        # 图像级叠加的 image 也需要放缩。
        if (src_h != img_h) or (src_w != img_w):
            self.img = cv2.resize(self.img, (src_w, src_h))
        return cv2.addWeighted(src_img, 1, self.img, self.weight, 1)


@dataclasses.dataclass
class DrawVisuAttrs(BaseVisuAttrs):
    color: Union[ColorTuple, str, None] = None
    thickness: Optional[int] = None
    sync_attributes: list[str] = dataclasses.field(
        default_factory=lambda: ["color", "thickness"])


@dataclasses.dataclass
class SquareColorPair(object):
    dot_pair: tuple[list[int], list[int]]
    color: Union[tuple[int, ...], ColorTuple, str, None] = None
    thickness: Optional[int] = None

    def sync(self, src: DrawRectVisu):
        """Sync all attributes from its father class.
        """
        assert isinstance(src, (self.__class__, DrawRectVisu)), (
            f"properties should be from {self.__class__.__name__}, " +
            f"got {src.__class__.__name__}.")
        if self.color is None:
            self.color = src.color
        if self.thickness is None:
            self.thickness = src.thickness


@dataclasses.dataclass
class DrawRectVisu(DrawVisuAttrs):
    pair_list: Union[list[SquareColorPair], LAZY_FLAG] = AS_INPUT

    def render(self, src_img: U8Mat, scaler: tuple[float, float]):
        if self.pair_list == AS_INPUT or self.thickness is None:
            self.not_ready_error_raiser()
        for pdata in self.pair_list:
            pdata.sync(self)
            if pdata.color is None:
                self.not_ready_error_raiser()
            pdata.color = parse_color(pdata.color)
            dot_pair0 = scale_pt(pdata.dot_pair[0], scaler)
            dot_pair1 = scale_pt(pdata.dot_pair[1], scaler)
            src_img = cv2.rectangle(src_img, dot_pair0, dot_pair1, pdata.color,
                                    self.thickness)
        return src_img


@dataclasses.dataclass
class DotColorPair(object):
    dot: tuple[int, int]
    color: Union[tuple[int, ...], ColorTuple, str, None]
    radius: Optional[int] = None

    def sync(self, src: DrawCircleVisu):
        """Sync all attributes from its father class.
        """
        assert isinstance(src, (self.__class__, DrawCircleVisu)), (
            f"properties should be from {self.__class__.__name__}, " +
            f"got {src.__class__.__name__}.")
        if self.color is None:
            self.color = src.color
        if self.radius is None:
            self.radius = src.radius


@dataclasses.dataclass
class DrawCircleVisu(DrawVisuAttrs):
    dot_list: Union[list[DotColorPair], LAZY_FLAG] = AS_INPUT
    radius: Union[int, None] = None
    sync_attributes: list[str] = dataclasses.field(
        default_factory=lambda: ["color", "thickness", "radius"])

    def render(self, src_img: U8Mat, scaler: tuple[float, float]):
        if self.dot_list == AS_INPUT or self.thickness is None:
            self.not_ready_error_raiser()
        for ddata in self.dot_list:
            ddata.sync(self)
            if ddata.color is None or ddata.radius is None:
                self.not_ready_error_raiser()
            dot_pair = scale_pt(ddata.dot, scaler)
            ddata.color = parse_color(ddata.color)
            src_img = cv2.circle(src_img, dot_pair, ddata.radius, ddata.color,
                                 self.thickness)
        return src_img


@dataclasses.dataclass
class TextColorPair(object):
    text: str
    position: Union[str, list[int], None] = None
    color: Union[ColorTuple, str, None] = None

    def sync(self, src: TextVisu):
        """Sync all attributes from its father class.
        """
        assert isinstance(src, (self.__class__, TextVisu)), (
            f"properties should be from {self.__class__.__name__}, " +
            f"got {src.__class__.__name__}.")
        if self.color is None:
            self.color = src.color
        if self.position is None:
            self.position = src.position


@dataclasses.dataclass
class TextVisu(BaseVisuAttrs):
    text_list: Union[list[TextColorPair], LAZY_FLAG] = AS_INPUT
    position: Union[list[int], str, LAZY_FLAG, None] = None
    color: Union[ColorTuple, str, LAZY_FLAG, None] = None
    font_face: Optional[int] = None
    font_scale: Optional[float] = None
    font_thickness: Optional[int] = None
    sync_attributes: list[str] = dataclasses.field(
        default_factory=lambda:
        ["position", "color", "font_face", "font_scale", "font_thickness"])

    def render(self, src_img: U8Mat, scaler: tuple[float, float]) -> U8Mat:
        if (self.text_list == AS_INPUT or self.font_face is None
                or self.font_scale is None or self.font_thickness is None):
            raise self.not_ready_error_raiser()
        for tdata in self.text_list:
            tdata.sync(self)
            if tdata.position is None or isinstance(
                    tdata.position, str) or tdata.color is None:
                raise self.not_ready_error_raiser()
            tdata.color = parse_color(tdata.color)
            put_pos = scale_pt(tdata.position, scaler)
            src_img = cv2.putText(src_img,
                                  tdata.text,
                                  org=put_pos,
                                  fontFace=self.font_face,
                                  fontScale=self.font_scale,
                                  color=tdata.color,
                                  thickness=self.font_thickness)
        return src_img


class OpenCVMetVisu(object):

    def __init__(self,
                 exp_time: float,
                 resolution: list[int],
                 flag: bool = True,
                 delay: int = DEFAULT_VISUAL_DELAY,
                 interrupt_key: str = DEFAULT_INTERRUPT_KEY,
                 visu_param_list: Optional[list[BaseVisuAttrs]] = None,
                 mor_thickness: int = 2,
                 font_size: float = 0.5,
                 font_color: Optional[str] = None,
                 font_thickness: int = 1,
                 font_gap: int = 20,
                 radius: int = 2,
                 dist2boarder: int = 10,
                 fontface: int = cv2.FONT_HERSHEY_COMPLEX) -> None:
        """基于OpenCV的流星检测可视化组件。

        Args:
            exp_time (int): exposure_time
            resolution (Union[tuple,list]): resolution of the Debug window.
            flag (bool, optional): Whether to show the debug window. Defaults to True.
            delay (int, optional): _description_. Defaults to DEFAULT_VISUAL_DELAY.
            interrupt_key (str, optional): _description_. Defaults to DEFAULT_INTERRUPT_KEY.
            visu_param_list
            mor_thickness (int, optional): _description_. Defaults to 2.
            font_size (float, optional): _description_. Defaults to 0.5.
            font_color (Optional[str], optional): _description_. Defaults to None.
            font_thickness (int, optional): _description_. Defaults to 1.
            font_gap (int, optional): _description_. Defaults to 20.
            dist2boarder (int, optional): _description_. Defaults to 10.
            fontface (_type_, optional): _description_. Defaults to cv2.FONT_HERSHEY_COMPLEX.
        """
        assert len(
            interrupt_key
        ) == 1, f"interrupt key should be a single key, but got {interrupt_key}."

        self.flag = flag
        self.visual_delay = int(exp_time * delay)
        self.resolution = resolution
        self.interrupt_key = ord(interrupt_key)
        self.font_size = font_size
        self.font_color = COLOR_MAP[font_color] if font_color else COLOR_MAP[
            DEFAULT_COLOR]
        self.font_thickness = font_thickness
        self.mor_thickness = mor_thickness
        self.font_gap = font_gap
        self.fontface = fontface
        self.radius = radius
        self.dist2boarder = dist2boarder
        self.manual_stop = False
        self.deafult_rect_config = DrawRectVisu("default",
                                                color=self.font_color,
                                                thickness=self.mor_thickness)
        self.default_text_config = TextVisu("default",
                                            position="as-input",
                                            color=self.font_color,
                                            font_face=self.fontface,
                                            font_scale=self.font_size,
                                            font_thickness=self.font_thickness)
        self.logger = get_default_logger()

        # 默认的接口：仅时间。（BG图像不通过该接口解析）
        # 还可以有：视频的基本信息。进度条。

        self.visu_param: list[
            BaseVisuAttrs] = visu_param_list if visu_param_list else []
        self.visu_param.insert(
            0, TextVisu(name="timestamp",
                        position="left-bottom",
                        color="white"))
        self.init_visu_params(self.visu_param)

        # 准备图像使用的Transform
        self.img_expand_channel = Transform()
        self.img_expand_channel.expand_3rd_channel(3)

    def init_visu_params(self, params: list[BaseVisuAttrs]):
        """可视化参数初始化器。

        Args:
            params (list[BaseVisuAttrs]): _description_

        Returns:
            list[dict]: _description_
        """
        self.img_visu_param: list[ImgVisuAttrs] = []
        self.draw_visu_param: list[DrawVisuAttrs] = []
        self.text_visu_param: list[TextVisu] = []
        img_w, img_h = self.resolution
        # 位置累加器，按顺序解析并累加默认位置
        text_pos_temp = {
            k: [
                int(w * img_w) + int(self.dist2boarder * dw),
                int(h * img_h) + int(self.dist2boarder * dh)
            ]
            for (k, (w, h, dw, dh, _, _)) in POSITION_MAP.items()
        }
        text_offset = {
            k: [int(ow * self.font_gap),
                int(oh * self.font_gap)]
            for (k, (_, _, _, _, ow, oh)) in POSITION_MAP.items()
        }
        # 主解析循环，填充默认值，简化绘制时的步骤。
        for obj in params:
            if not isinstance(obj, (ImgVisuAttrs, DrawVisuAttrs, TextVisu)):
                self.logger.warning(
                    f"Unrecognized visu type: {obj.__class__.__name__}. " +
                    "Ignore to continue...")
                continue
            if isinstance(obj, ImgVisuAttrs):
                self.img_visu_param.append(obj)
                continue
            # 对于绘制类和文字类的注册项，位置需要明确定义
            # 但因为复用dataclass的初始化函数，定义时对此没有检查，仅有运行时检查。
            if isinstance(obj, DrawVisuAttrs):
                if isinstance(obj, DrawRectVisu):
                    obj.sync(self.deafult_rect_config)
                self.draw_visu_param.append(obj)
            else:
                obj.sync(self.default_text_config)
                # 固定位置解析
                if obj.position in POSITION_MAP:
                    pos = obj.position
                    # 更新临时位置后，将当前固定位置转换为坐标。
                    text_pos_temp[pos] = pt_offset(text_pos_temp[pos],
                                                   text_offset[pos])
                    obj.position = text_pos_temp[pos]
                self.text_visu_param.append(obj)

    def display_a_frame(self, base_img: U8Mat,
                        data_list: list[BaseVisuAttrs]) -> bool:
        """使用传入参数渲染完整的一帧。

        Args:
            data (dict): _description_

        Returns:
            bool: 状态值，代表是否成功渲染并展示。
        """
        # 通过键盘中断时返回失败信号
        # 如果不渲染则固定返回成功
        if not self.flag:
            return True

        if (cv2.waitKey(self.visual_delay) & 0xff == self.interrupt_key):
            self.manual_stop = True
            return False

        data = {d.name: d for d in data_list}

        ### 基础图像预处理
        # 转换灰度图像为BGR图像。
        if len(base_img.shape) == 2:
            base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        # 放缩到可视化分辨率
        scaler = (1, 1)
        if self.resolution[0] != base_img.shape[1] or self.resolution[
                1] != base_img.shape[0]:
            scaler = (base_img.shape[0] / self.resolution[1],
                      base_img.shape[1] / self.resolution[0])
            base_img = cv2.resize(base_img, self.resolution)
        # 渲染顺序：优先所有img，然后绘图，最后是text
        # img: 仅支持在背景上继续叠加。
        for img_visu in self.img_visu_param:
            key = img_visu.name
            # 跳过不渲染（或缺失）的值
            # TODO: 检查一下：是否所有都必须要在data中带值？存在一定展示的项，不需要存在于data中吗?下同。
            if not key in data:
                continue
            runtime_visu = data[key]
            runtime_visu.sync(img_visu)
            base_img = runtime_visu.render(base_img, scaler)
        # 绘图类操作
        # 工作流高度相似的现在，是不是还可以进一步简化一下...
        for draw_visu in self.draw_visu_param:
            key = draw_visu.name
            # 跳过不渲染（或缺失）的值
            # TODO: 检查一下：是否所有都必须要在data中带值？存在一定展示的项，不需要存在于data中吗
            if not key in data:
                continue
            runtime_draw = data[key]
            runtime_draw.sync(draw_visu)
            base_img = runtime_draw.render(base_img, scaler)

        for text_visu in self.text_visu_param:
            key = text_visu.name
            # 跳过不渲染（或缺失）的值
            # TODO: 检查一下：是否所有都必须要在data中带值？存在一定展示的项，不需要存在于data中吗
            if not key in data:
                continue
            runtime_text = data[key]
            runtime_text.sync(text_visu)
            base_img = runtime_text.render(base_img, scaler)

        # TODO: Add "save_sample_img / save_detect_video" function for debug.
        cv2.imshow(
            f"Debug Window (Press {chr(self.interrupt_key).upper()} to exit)",
            base_img)
        return True

    def stop(self):
        """销毁时动作。
        """
        cv2.destroyAllWindows()
