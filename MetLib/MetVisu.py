"""管理关于可视化的绘图句柄及API。
暂定：DEBUG模式会打印详细日志，并默认启用可视化工具。
如果禁用可视化工具需要额外在命令行指定。
当启用可视化模式时
"""
import cv2
import numpy as np
from easydict import EasyDict
from typing import Optional, Callable, Union
from copy import deepcopy
from .utils import pt_offset

DEFAULT_VISUAL_DELAY = 200
DEFAULT_INTERRUPT_KEY = "q"
DEFAULT_COLOR = "white"

COLOR_MAP = {
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "white": (255, 255, 255),
    "purple": (128, 64, 128),
    "black": (0, 0, 0)
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
                 mor_thickness=2,
                 font_size: float = 0.5,
                 font_color: Optional[str] = None,
                 font_thickness: int = 1,
                 font_gap: int = 20,
                 dist2boarder: int = 10,
                 fontface=cv2.FONT_HERSHEY_COMPLEX) -> None:
        """基于OpenCV的流星检测可视化类。

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

        self.interrupt_key = ord(interrupt_key)
        self.font_size = font_size
        self.font_color = COLOR_MAP[font_color] if font_color else COLOR_MAP[
            DEFAULT_COLOR]
        self.font_thickness = font_thickness
        self.mor_thickness = mor_thickness
        self.font_gap = font_gap
        self.fontface = fontface
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

        self.img_w, self.img_h = resolution
        self.init_text_pos = {
            k: [
                int(w * self.img_w) + self.dist2boarder * dw,
                int(h * self.img_h) + self.dist2boarder * dh
            ]
            for (k, (w, h, dw, dh, _, _)) in POSITION_MAP.items()
        }
        self.text_offset = {
            k: [ow * self.font_gap, oh * self.font_gap]
            for (k, (_, _, _, _, ow, oh)) in POSITION_MAP.items()
        }

    def display_a_frame(self, data: dict) -> bool:
        # 通过键盘中断时返回失败信号
        # 如果不渲染则固定返回成功
        if not self.flag:
            return True

        if (cv2.waitKey(self.visual_delay) & 0xff == self.interrupt_key):
            self.manual_stop = True
            return False

        # 渲染图像并通过OpenCV句柄展示
        rendered_img = self.rend_frame(data)

        # TODO: Add "save_sample_img" function for debug.
        cv2.imshow(
            f"Debug Window (Press {chr(self.interrupt_key).upper()} to exit)",
            rendered_img)
        return True

    def stop(self):
        """销毁时动作。
        """
        cv2.destroyAllWindows()

    def rend_frame(self, data):
        """_summary_

        Args:
            data (dict): data is an Easydict object that is used to generate everything.
            the following parameters should be provided: 
                bg: Union[np.ndarray, Callable[[],np.ndarray]]
                info: List[dict(type: str, pos: Union[str, tuple, list], cfg: dict)]
        Returns:
            np.ndarray: _description_
        """
        # 渲染帧背景
        if isinstance(data["bg"], Callable):
            bg_img = data["bg"]()
        else:
            bg_img = data["bg"]
        # 位置累加器
        text_pos_temp = deepcopy(self.init_text_pos.copy())

        # 绘图/压字性质操作
        for (info_type, pos, cfg) in data["info"]:
            # 解析并替换cfg中的固定字段，如color
            if isinstance(cfg.get("color", None), str):
                cfg["color"] = COLOR_MAP[cfg["color"]]

            if info_type == "rectangle":
                pt1, pt2 = pos
                bg_img = cv2.rectangle(bg_img,
                                       pt1,
                                       pt2,
                                       color=cfg.get("color", self.font_color),
                                       thickness=cfg.get(
                                           "thickness", self.mor_thickness))
            elif info_type == "text":
                if isinstance(pos, str):
                    text_pos_temp[pos] = pt_offset(text_pos_temp[pos],
                                                   self.text_offset[pos])
                    pos = text_pos_temp[pos]
                bg_img = cv2.putText(bg_img,
                                     cfg["text"],
                                     org=pos,
                                     fontFace=cfg.get("foneFace",
                                                      self.fontface),
                                     fontScale=cfg.get("fontScale",
                                                       self.font_size),
                                     color=cfg.get("color", self.font_color),
                                     thickness=cfg.get("thickness",
                                                       self.font_thickness))
        return bg_img