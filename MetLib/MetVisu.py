"""管理关于可视化的绘图句柄及API。
VISU模式下会启用可视化。

可视化接口的设计
detector和collector需要在参数中包含visu_param，并支持visu方法。

visu_param是一个dict[str, list]的对象。
对于每一项，其key为名称，value为包含两个参数的list。
list的第一项代表可视化类型，目前支持"draw"(绘制),"img"（图像叠加），"text"（文字）。
第二项代表对应可视化需要的参数。

draw的参数如下：
    - type 代表需要绘制的图像。支持 "rectangle" 与 "circle" 。
    - color 代表需要绘制的颜色。支持字符串形式的颜色名称，"as-input" 或者具体的RGB值。
    - 绘制不同类型时还需要提供的参数不同。这些参数都支持 "as-input"，即在运行时指定位置。
    - circle类还需要：radius，position，thickness。
    - rectangle类还需要：position，thickness，

img的参数如下：
    - weight 叠加的强度。
    如果希望叠加灰度图像，需要指定 color 叠加的颜色。

text的参数如下：
    - position 文字输出的位置。对于需要监视的数值，可以通过使用默认的位置，MetVisu会自动排版。
    - color 代表绘制的颜色。


visu()方法返回一个dict[str, list]的对象。
对于每一项，其key为名称，value为需要按照规则可视化的list。
注意：固定位置的text不支持长度大于1的list的（其位置需要在输入时确定，因此不能接受变长列表，但需要是列表（以统一输入格式）。目前没有做对应检查。）

"""
import cv2
import numpy as np
from typing import Optional, Union
from .utils import pt_offset, Transform, gray2colorimg
from .MetLog import get_default_logger

DEFAULT_VISUAL_DELAY = 200
DEFAULT_INTERRUPT_KEY = "q"
DEFAULT_COLOR = "white"

COLOR_MAP = {
    "black": (0, 0, 0),
    "green": (0, 255, 0),
    "orange": (0, 128, 255),
    "purple": (128, 64, 128),
    "red": (0, 0, 255),
    "white": (255, 255, 255),
    "yellow": (0, 255, 255)
}

# 描述初始位置
# TODO: 稳健性需要更多测试
# in [w,h,ow,oh,dw,dh]
POSITION_MAP = {
    "left": (0, 0.5, 1, 0, 0, 1),
    "left-top": (0, 0, 1, 1, 0, 1),
    "top": (0.5, 0, 0, 1, 0, 1),
    "left-bottom": (0, 1, 1, 0, 0, -1),
    "right-top": (0.8, 0, 0, 1, 0, 1),
    "right": (0.8, 0.5, 0, 0, 0, 1),
    "right-bottom": (0.8, 0.9, 0, 0, 0, -1),
    "bottom": (0.5, 0.9, 0, -1, 0, -1)
}


class OpenCVMetVisu(object):

    def __init__(self,
                 exp_time: int,
                 resolution: Union[tuple, list],
                 flag: bool = True,
                 delay: int = DEFAULT_VISUAL_DELAY,
                 interrupt_key: str = DEFAULT_INTERRUPT_KEY,
                 visu_param_list: Optional[list] = None,
                 mor_thickness=2,
                 font_size: float = 0.5,
                 font_color: Optional[str] = None,
                 font_thickness: int = 1,
                 font_gap: int = 20,
                 radius: int = 2,
                 dist2boarder: int = 10,
                 fontface=cv2.FONT_HERSHEY_COMPLEX) -> None:
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
        self.default_param = {
            "rectangle":
            dict(color=self.font_color, thickness=self.mor_thickness),
            "circle":
            dict(),
            "text":
            dict(fontFace=self.fontface,
                 fontScale=self.font_size,
                 color=self.font_color,
                 thickness=self.font_thickness)
        }
        self.logger = get_default_logger()

        # 默认的接口：仅时间。（BG图像不通过该接口解析）
        # 还可以有：视频的基本信息。进度条。
        self.visu_param = dict(
            timestamp=["text", {
                "position": "left-bottom",
                "color": "white"
            }])
        if visu_param_list is not None:
            for other_params in visu_param_list:
                self.visu_param.update(other_params)
        self.init_visu_params(self.visu_param)

        # 准备图像使用的Transform
        self.img_gray2bgr = Transform()
        self.img_gray2bgr.opencv_GRAY2BGR()
        self.img_expand_channel = Transform()
        self.img_expand_channel.expand_3rd_channel(3)

    def init_visu_params(self, params):
        """可视化参数初始化器。

        Args:
            params (_type_): _description_

        Returns:
            list[dict]: _description_
        """
        self.img_visu_param, self.draw_visu_param, self.text_visu_param = dict(
        ), dict(), dict()
        # 需要在该步骤填充默认值，简化绘制时的步骤。
        # 位置累加器，按顺序解析并累加默认位置
        img_w, img_h = self.resolution
        text_pos_temp = {
            k: [
                int(w * img_w) + self.dist2boarder * dw,
                int(h * img_h) + self.dist2boarder * dh
            ]
            for (k, (w, h, dw, dh, _, _)) in POSITION_MAP.items()
        }
        text_offset = {
            k: [ow * self.font_gap, oh * self.font_gap]
            for (k, (_, _, _, _, ow, oh)) in POSITION_MAP.items()
        }
        # 主解析循环
        for key, (visu_type, cfg) in params.items():
            if visu_type == "img":
                self.img_visu_param[key] = cfg
            elif visu_type == "draw":
                cfg["color"] = cfg.get("color", self.font_color)
                cfg["thickness"] = cfg.get("thickness", self.mor_thickness)
                if cfg["type"] == "circle":
                    cfg["radius"] = cfg.get("radius", self.radius)
                self.draw_visu_param[key] = cfg
            elif visu_type == "text":
                cfg["color"] = cfg.get("color", self.font_color)
                # 解析及更新固定位置
                if cfg["position"] in POSITION_MAP:
                    pos = cfg["position"]
                    text_pos_temp[pos] = pt_offset(text_pos_temp[pos],
                                                   text_offset[pos])
                    cfg["position"] = text_pos_temp[pos]
                cfg["fontFace"] = cfg.get("foneFace", self.fontface)
                cfg["fontScale"] = cfg.get("fontScale", self.font_size)
                cfg["thickness"] = cfg.get("thickness", self.font_thickness)
                self.text_visu_param[key] = cfg

    def display_a_frame(self, data: dict) -> bool:
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

        # 渲染图像并通过OpenCV句柄展示
        rendered_img = self.rend_frame(data)

        # TODO: Add "save_sample_img / save_detect_video" function for debug.
        cv2.imshow(
            f"Debug Window (Press {chr(self.interrupt_key).upper()} to exit)",
            rendered_img)
        return True

    def stop(self):
        """销毁时动作。
        """
        cv2.destroyAllWindows()

    def rend_frame(self, data: dict) -> np.ndarray:
        """_summary_

        Args:
            data (dict): data is an dict object that is used to generate everything.
            the following parameters should be provided: 
                Every fmt should be like [value_type, type, cfg: dict]
            支持的render_type: 
                draw - 绘制
                text - 文本
                img - 图像
                
        Returns:
            np.ndarray: _description_
        """

        # 首位为背景图像
        assert "main_bg" in data, "base image required."
        base_img = data["main_bg"]
        # 转换灰度图像为BGR图像。
        if len(base_img.shape) == 2:
            base_img = self.img_gray2bgr.exec_transform(base_img)
        # 渲染顺序：优先所有img，然后绘图，最后是text
        # img: 仅支持在背景上继续叠加。
        for key, base_cfg in self.img_visu_param.items():
            # 跳过不渲染（或缺失）的值
            if not key in data:
                continue
            for cfg in data[key]:
                img = cfg["img"]
                self.fill_cfg_w_default(cfg, base_cfg)
                if len(img.shape) == 2:
                    if "color" in cfg:
                        img = gray2colorimg(
                            img,
                            np.array(self.parse_color(cfg["color"]),
                                     dtype=np.uint8).reshape((1, -1)))
                    else:
                        img = self.img_gray2bgr.exec_transform(img)
                base_img = cv2.addWeighted(base_img, 1, img, cfg["weight"], 1)
        # 绘图类操作
        for key, base_cfg in self.draw_visu_param.items():
            # 跳过不渲染（或缺失）的值
            if not key in data:
                continue
            for cfg in data[key]:
                self.fill_cfg_w_default(cfg, base_cfg)
                if cfg["type"] == "rectangle":
                    pt1, pt2 = cfg["position"]
                    base_img = cv2.rectangle(base_img, pt1, pt2,
                                             self.parse_color(cfg["color"]),
                                             cfg["thickness"])
                elif cfg["type"] == "circle":
                    pt = cfg["position"]
                    base_img = cv2.circle(base_img, pt, cfg["radius"],
                                          self.parse_color(cfg["color"]),
                                          cfg["thickness"])

        # 压字类操作
        for key, base_cfg in self.text_visu_param.items():
            # 跳过不渲染（或缺失）的值
            if not key in data:
                continue
            for cfg in data[key]:
                # 对于固定位置的，应当在输入时已确定唯一渲染位置；
                # 对于不确定位置的，不允许在运行时使用对应可变位置。<-没有检查
                self.fill_cfg_w_default(cfg, base_cfg)
                base_img = cv2.putText(base_img,
                                       cfg["text"],
                                       org=cfg["position"],
                                       fontFace=cfg["fontFace"],
                                       fontScale=cfg["fontScale"],
                                       color=self.parse_color(cfg["color"]),
                                       thickness=cfg["thickness"])

        # 放缩到指定的分辨率
        # TODO: 会导致字体类看不清楚...
        # h, w, _ = base_img.shape
        # tgt_w, tgt_w = self.resolution
        if self.resolution[0]!=base_img.shape[1] or self.resolution[1]!=base_img.shape[0]:
            base_img = cv2.resize(base_img, self.resolution)

        return base_img

    def fill_cfg_w_default(self, cfg, default_cfg):
        """将default_cfg的值填充到cfg中，构建完整的参数列表。
        TODO: 如果default_cfg中有as_input的值没有在cfg中被填充，则抛出警告并忽略对应可视化。
        Args:
            cfg (_type_): _description_
            default_cfg (_type_): _description_
        """
        for key, value in default_cfg.items():
            if key in cfg: continue
            if value == "as_input":
                raise ValueError(f"Defined \"as_input\" for {key}, "
                                 f"which is not found in visu_input.")
            cfg.update({key: value})
        return cfg

    def parse_color(self, color):
        if isinstance(color, str):
            color = COLOR_MAP[color]
        assert len(color) == 3, "invalid color"
        return color
