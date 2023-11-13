"""管理关于可视化的绘图句柄及API。
暂定：DEBUG模式会打印详细日志，并默认启用可视化工具。
如果禁用可视化工具需要额外在命令行指定。
当启用可视化模式时
"""
import cv2
import numpy as np
from easydict import EasyDict
from typing import Optional, Callable

DEFAULT_VISUAL_DELAY = 400
DEFAULT_INTERRUPT_KEY = "q"
DEFAULT_COLOR = "white"

COLOR_MAP = {
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "white": (255, 255, 255),
    "purple": (128,64,128)
}

# in [w,h]
POSITION_MAP = {
    "left": (0, 0.5),
    "left-top": (0, 0),
    "top": (0.5, 0),
    #    "right-top": (1, 0),
    #    "right": (1, 0.5),
    #    "right-bottom": (1, 1),
    #    "bottom": (0.5, 1),
    #    "left-bottom": (0, 1)
}


class OpenCVMetVisu(object):
    # 支持定义操作行为
    rectangle=cv2.rectangle
    circle = cv2.circle

    def __init__(self,
                 exp_time: int,
                 flag: bool = True,
                 delay: int = DEFAULT_VISUAL_DELAY,
                 interrupt_key: str = DEFAULT_INTERRUPT_KEY,
                 mor_thickness= 2,
                 font_size: float = 0.5,
                 font_color: Optional[str] = None,
                 font_thickness: int = 1,
                 font_gap: int = 20,
                 dist2boarder: int = 10,
                 fontface=cv2.FONT_HERSHEY_COMPLEX) -> None:
        self.flag = flag
        self.visual_delay = int(exp_time*delay)
        self.interrupt_key = ord(interrupt_key)
        self.font_size = font_size
        self.font_color = COLOR_MAP[font_color] if font_color else COLOR_MAP[
            DEFAULT_COLOR]
        self.font_thickness = font_thickness
        self.font_gap = font_gap
        self.fontface = fontface
        self.dist2boarder = dist2boarder
        self.text_pos_dict = {
            k: [w, h, int(1 - 2 * w), int(1 - 2 * h)]
            for k, (w, h) in POSITION_MAP.items()
        }

    def display_a_frame(self, data) -> bool:
        # 通过键盘中断时返回失败信号
        # 如果不渲染则固定返回成功
        if (cv2.waitKey(self.visual_delay) & 0xff == self.interrupt_key):
            return False
        if not self.flag:
            return True
        
        # 渲染图像并通过OpenCV句柄展示
        rendered_img = self.rend_frame(data)
        # TODO: Add "save_sample_img" function for debug.
        #cv2.imwrite("test/frame_%s.jpg"%i,draw_img)
        cv2.imshow(
            f"Debug Window (Press {chr(self.interrupt_key).upper()} to exit)",
            rendered_img)
        return True

    def destroy(self):
        """销毁时动作。
        """
        cv2.destroyAllWindows()

    def rend_frame(self, data):
        """_summary_

        Args:
            data (dict): data is an Easydict object that is used to generate everything.
            the following parameters should be provided: 
                bg: Union[np.ndarray, Callable[[],np.ndarray]]
                info: List[dict(text: str, pos: str, cfg: dict)]
        Returns:
            np.ndarray: _description_
        """
        # 渲染帧背景
        if isinstance(data.bg, Callable):
            bg_img = data.bg()
        else:
            bg_img = data.bg

        # 绘图性质操作


        # 压字
        img_h, img_w = bg_img.shape()[:2]
        text_pos_temp = {
            k: [
                int(w * img_w) + self.dist2boarder * dw,
                int(h * img_h) + self.dist2boarder * dh
            ]
            for (k, (w, h, dw, dh)) in self.text_pos_dict.items()
        }
        for (text, pos, cfg) in data.texts:
            if isinstance(pos, str):
                text_pos = text_pos_temp[pos]
            elif isinstance(pos, (tuple,list)):
                text_pos = pos
        
        return bg_img