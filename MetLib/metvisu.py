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

from .imgproc import Transform
from .metlog import get_default_logger
from .utils import COLOR_MAP, U8Mat, pt_offset

DEFAULT_VISUAL_DELAY = 200
DEFAULT_INTERRUPT_KEY = "q"
DEFAULT_COLOR = "white"

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
    color: Union[ColorTuple, str, None] = None
    sync_attributes: list[str] = dataclasses.field(
        default_factory=lambda: ['weight', 'img', 'color'])

    def render(self, src_img: U8Mat, scaler: tuple[float, float]) -> U8Mat:
        if self.weight is None or self.img is None:
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
    dot_pair: Optional[tuple[list[int], list[int]]] = None
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
    pair_list: Optional[list[SquareColorPair]] = None

    def render(self, src_img: U8Mat, scaler: tuple[float, float]):
        if self.thickness is None:
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
    dot: Optional[tuple[int, int]] = None
    color: Union[tuple[int, ...], ColorTuple, str, None] = None
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
    dot_list: Optional[list[DotColorPair]] = None
    radius: Union[int, None] = None
    sync_attributes: list[str] = dataclasses.field(
        default_factory=lambda: ["color", "thickness", "radius"])

    def render(self, src_img: U8Mat, scaler: tuple[float, float]):
        if self.thickness is None:
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
    text: Optional[str] = None
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
    text_list: Optional[list[TextColorPair]] = None
    position: Union[list[int], str, LAZY_FLAG, None] = None
    color: Union[ColorTuple, str, LAZY_FLAG, None] = None
    font_face: Optional[int] = None
    font_scale: Optional[float] = None
    font_thickness: Optional[int] = None
    position_flag: bool = False
    sync_attributes: list[str] = dataclasses.field(
        default_factory=lambda:
        ["position", "color", "font_face", "font_scale", "font_thickness"])

    def render(self, src_img: U8Mat, scaler: tuple[float, float]) -> U8Mat:
        if (self.font_face is None or self.font_scale is None
                or self.font_thickness is None):
            raise self.not_ready_error_raiser()
        for tdata in self.text_list:
            tdata.sync(self)
            if tdata.position is None or isinstance(
                    tdata.position, str) or tdata.color is None:
                raise self.not_ready_error_raiser()
            tdata.color = parse_color(tdata.color)
            if not self.position_flag:
                put_pos = scale_pt(tdata.position, scaler)
            else:
                put_pos = tdata.position
            src_img = cv2.putText(src_img,
                                  tdata.text,
                                  org=put_pos,
                                  fontFace=self.font_face,
                                  fontScale=self.font_scale,
                                  color=tdata.color,
                                  thickness=self.font_thickness)
        return src_img


# 4个角落的锚点位置（占画面比例）和图表展开方向
# key: corner名称, value: (anchor_x比例, anchor_y比例, 展开方向x, 展开方向y)
# 展开方向 1=向右/向下, -1=向左/向上
CHART_CORNER_MAP: dict[str, tuple[float, float, int, int]] = {
    "left-top": (0, 0, 1, 1),
    "right-top": (1, 0, -1, 1),
    "left-bottom": (0, 1, 1, -1),
    "right-bottom": (1, 1, -1, -1),
}

CHART_PADDING = 5       # 图表上/右/下内边距（像素）
CHART_PADDING_LEFT = 38 # 图表左侧内边距，留给 Y 轴标注
CHART_MARGIN = 8        # 图表与画面边缘的距离（像素）
CHART_FONT_SCALE = 0.7
CHART_FONT_FACE = cv2.FONT_HERSHEY_PLAIN


@dataclasses.dataclass
class TimeSeriesChartVisu(BaseVisuAttrs):
    """调用方每帧传入的轻量更新包。

    结构参数（仅首次注册时生效）：corner, chart_w, chart_h, max_points
    渲染参数（每帧读取）：y_min, y_max, label, line_color, bg_alpha
    """
    current_value: float = 0.0
    # 结构参数
    corner: str = "right-bottom"
    chart_w: int = 200
    chart_h: int = 100
    max_points: int = 100
    # 渲染参数
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    label: Optional[str] = None
    line_color: Union[ColorTuple, str] = "green"
    bg_alpha: float = 0.5

    def render(self, src_img: U8Mat, scaler: tuple[float, float]) -> U8Mat:  # noqa: ARG002
        raise RuntimeError(
            "TimeSeriesChartVisu is a lightweight update packet and cannot render directly. "
            "Use OpenCVMetVisu to manage chart rendering via the registry.")


@dataclasses.dataclass
class TimeSeriesChartHandle(object):
    """由 OpenCVMetVisu 内部持有的有状态图表实例，负责维护历史缓冲区并执行渲染。"""
    name: str
    corner: str
    chart_w: int
    chart_h: int
    max_points: int

    _buffer: list[float] = dataclasses.field(default_factory=lambda: [], init=False)

    def push(self, value: float):
        self._buffer.append(value)
        if len(self._buffer) > self.max_points:
            self._buffer.pop(0)

    def render(self, src_img: U8Mat, visu: TimeSeriesChartVisu) -> U8Mat:
        """在 src_img 上就地绘制时序曲线图，返回绘制后的图像。

        src_img 已经是可视化分辨率，直接以像素为单位操作。
        """
        if len(self._buffer) < 2:
            return src_img

        img_h, img_w = src_img.shape[:2]
        w, h = self.chart_w, self.chart_h

        # --- 确定图表左上角像素坐标 ---
        if self.corner not in CHART_CORNER_MAP:
            return src_img
        ax, ay, dx, dy = CHART_CORNER_MAP[self.corner]
        anchor_x = int(ax * img_w)
        anchor_y = int(ay * img_h)
        # dx/dy 决定图表相对锚点的展开方向
        x0 = anchor_x + CHART_MARGIN if dx > 0 else anchor_x - CHART_MARGIN - w
        y0 = anchor_y + CHART_MARGIN if dy > 0 else anchor_y - CHART_MARGIN - h
        x0 = max(0, min(x0, img_w - w))
        y0 = max(0, min(y0, img_h - h))
        x1, y1 = x0 + w, y0 + h

        # --- 绘制半透明背景 ---
        line_color = parse_color(visu.line_color)
        alpha = max(0.0, min(1.0, visu.bg_alpha))
        overlay = src_img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, src_img, 1 - alpha, 0, src_img)

        # --- 绘制边框 ---
        cv2.rectangle(src_img, (x0, y0), (x1, y1), line_color, 1)

        # --- 计算绘图区域（内边距，左侧加宽留给 Y 轴标注） ---
        px0 = x0 + CHART_PADDING_LEFT
        py0 = y0 + CHART_PADDING
        px1 = x1 - CHART_PADDING
        py1 = y1 - CHART_PADDING
        plot_w = px1 - px0
        plot_h = py1 - py0
        if plot_w <= 0 or plot_h <= 0:
            return src_img

        # --- 确定 Y 轴范围 ---
        y_min = visu.y_min if visu.y_min is not None else float(min(self._buffer))
        y_max = visu.y_max if visu.y_max is not None else float(max(self._buffer))
        if y_max == y_min:
            y_max = y_min + 1.0   # 防止除零

        # --- 将历史数据映射到像素坐标 ---
        n = len(self._buffer)
        pts: list[tuple[int, int]] = []
        for i, val in enumerate(self._buffer):
            px = px0 + int(i / (n - 1) * plot_w) if n > 1 else px0
            norm = (float(val) - y_min) / (y_max - y_min)
            norm = max(0.0, min(1.0, norm))
            py = py1 - int(norm * plot_h)   # y轴向下，值大则像素坐标小
            pts.append((px, py))

        # --- 绘制折线 ---
        for i in range(len(pts) - 1):
            cv2.line(src_img, pts[i], pts[i + 1], line_color, 1, cv2.LINE_AA)

        # --- 绘制 Y 轴 ymax/ymin 标注（绘图区左侧外沿） ---
        label_x = x0 + CHART_PADDING
        ymax_str = f"{y_max:.3g}"
        ymin_str = f"{y_min:.3g}"
        cv2.putText(src_img, ymax_str, (label_x, py0 + 9),
                    CHART_FONT_FACE, CHART_FONT_SCALE, line_color, 1, cv2.LINE_AA)
        cv2.putText(src_img, ymin_str, (label_x, py1),
                    CHART_FONT_FACE, CHART_FONT_SCALE, line_color, 1, cv2.LINE_AA)

        # --- 绘制标签（图表顶部边框上方居中，或右上角内） ---
        label: str = visu.label if visu.label is not None else self.name
        if label:
            cv2.putText(src_img, label,
                        (px0, y0 + CHART_PADDING + 9),
                        CHART_FONT_FACE, CHART_FONT_SCALE, line_color, 1, cv2.LINE_AA)

        return src_img


class OpenCVMetVisu(object):

    def __init__(self,
                 exp_time: float,
                 resolution: list[int],
                 flag: bool = True,
                 delay: int = DEFAULT_VISUAL_DELAY,
                 interrupt_key: str = DEFAULT_INTERRUPT_KEY,
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
        self.logger = get_default_logger()

        # 时序图表注册表：name -> TimeSeriesChartHandle
        self._chart_registry: dict[str, TimeSeriesChartHandle] = {}

        # 准备图像使用的Transform
        self.img_expand_channel = Transform()
        self.img_expand_channel.expand_3rd_channel(3)

    def display_a_frame(self, base_img: U8Mat,
                        data_list: list[BaseVisuAttrs]) -> bool:
        """使用传入参数渲染完整的一帧。

        Args:
            data (dict): _description_

        Returns:
            bool: 状态值，代表是否成功渲染并展示。
        """

        # 固定位置文字的“多条错开”累加器：
        # 旧版 init_visu_params 对同一 POSITION_MAP key 会按顺序累加 font_gap 偏移。
        # 这里恢复这一行为，保证渲染效果一致。
        img_w, img_h = self.resolution
        text_pos_temp = {
            k: [
                int(w * img_w) + int(self.dist2boarder * dw),
                int(h * img_h) + int(self.dist2boarder * dh),
            ]
            for (k, (w, h, dw, dh, _, _)) in POSITION_MAP.items()
        }
        text_offset = {
            k: [int(ow * self.font_gap), int(oh * self.font_gap)]
            for (k, (_, _, _, _, ow, oh)) in POSITION_MAP.items()
        }

        def _resolve_text_position(text: TextVisu):
            """把 POSITION_MAP 的字符串位置解析为像素坐标。"""
            if isinstance(text.position,
                          str) and text.position in POSITION_MAP:
                # 与旧版一致：先累加，再返回（保证“第一条”也有 1 * font_gap 偏移）。
                pos_key = text.position
                text_pos_temp[pos_key] = pt_offset(
                    text_pos_temp[pos_key], text_offset[pos_key])
                return text_pos_temp[pos_key]
            return text.position

        def _fill_text_defaults(text: TextVisu):
            text.font_face = self.fontface if text.font_face is None else text.font_face
            text.font_scale = self.font_size if text.font_scale is None else text.font_scale
            text.font_thickness = self.font_thickness if text.font_thickness is None else text.font_thickness
            text.color = self.font_color if text.color is None else text.color
            text.position = _resolve_text_position(text)

        def _fill_draw_defaults(draw: DrawVisuAttrs):
            # DrawRectVisu / DrawCircleVisu 的 dataclass 在 render 时依赖 thickness/radius/color 等字段。
            draw.thickness = self.mor_thickness if draw.thickness is None else draw.thickness
            draw.color = self.font_color if draw.color is None else draw.color
            if isinstance(draw, DrawCircleVisu):
                draw.radius = self.radius if draw.radius is None else draw.radius

        def _fill_img_defaults(img: ImgVisuAttrs):
            if img.weight is None:
                img.weight = 1.0

        # 通过键盘中断时返回失败信号
        # 如果不渲染则固定返回成功
        if not self.flag:
            return True

        if (cv2.waitKey(self.visual_delay) & 0xff == self.interrupt_key):
            self.manual_stop = True
            return False

        # 转换灰度图像为BGR图像。
        if len(base_img.shape) == 2:
            base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        # 放缩到可视化分辨率
        scaler = (1, 1)
        if self.resolution[0] != base_img.shape[1] or self.resolution[
                1] != base_img.shape[0]:
            scaler = (base_img.shape[1] / self.resolution[0],
                      base_img.shape[0] / self.resolution[1])
            base_img = cv2.resize(base_img, self.resolution)

        # 渲染顺序：img -> chart -> draw -> text
        img_list: list[ImgVisuAttrs] = []
        chart_list: list[TimeSeriesChartVisu] = []
        draw_list: list[DrawVisuAttrs] = []
        text_list: list[TextVisu] = []
        for obj in data_list:
            if isinstance(obj, ImgVisuAttrs):
                img_list.append(obj)
            elif isinstance(obj, TimeSeriesChartVisu):
                chart_list.append(obj)
            elif isinstance(obj, DrawVisuAttrs):
                draw_list.append(obj)
            elif isinstance(obj, TextVisu):
                text_list.append(obj)
            else:
                self.logger.warning(
                    f"Unrecognized visu type: {obj.__class__.__name__}. Ignore."
                )

        # img: 仅支持在背景上继续叠加。
        for img_visu in img_list:
            _fill_img_defaults(img_visu)
            base_img = img_visu.render(base_img, scaler)

        # chart: 首次注册，后续 push 新值并渲染。
        for chart_visu in chart_list:
            name = chart_visu.name
            if name not in self._chart_registry:
                self._chart_registry[name] = TimeSeriesChartHandle(
                    name=name,
                    corner=chart_visu.corner,
                    chart_w=chart_visu.chart_w,
                    chart_h=chart_visu.chart_h,
                    max_points=chart_visu.max_points,
                )
            handle = self._chart_registry[name]
            handle.push(chart_visu.current_value)
            base_img = handle.render(base_img, chart_visu)

        for draw_visu in draw_list:
            _fill_draw_defaults(draw_visu)
            base_img = draw_visu.render(base_img, scaler)

        for text_visu in text_list:
            _fill_text_defaults(text_visu)
            base_img = text_visu.render(base_img, scaler)

        # TODO: Add "save_sample_img / save_detect_video" function for debug.
        cv2.imshow(
            f"Debug Window (Press {chr(self.interrupt_key).upper()} to exit)",
            base_img)
        return True

    def stop(self):
        """销毁时动作。
        """
        cv2.destroyAllWindows()
