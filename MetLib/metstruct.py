"""
metstruct 定义 MetDetPy 使用的结构化数据和相关解析方法。
借助 dacite，可以容易的实现配置和结构化数据的解析。
"""

import dataclasses
import datetime
from typing import Any, Optional, Union

from dacite import from_dict


@dataclasses.dataclass
class Box(object):
    """A detection box.
    
    Order is required(x1<=x2, y1<=y2).
    """
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_list(cls, coord_list: list[int]):
        """将xyxy的list(可能乱序)转换为xyxy形式的坐标。

        Args:
            coord_list (list[int]): xyxy的list
        """
        assert len(
            coord_list
        ) == 4, f"Invalid coord list length: expect 4, got {len(coord_list)}."
        (x1, y1, x2, y2) = coord_list
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return cls(x1, y1, x2, y2)

    @classmethod
    def from_pts(cls, pt1: list[int], pt2: list[int]):
        assert len(pt1) == len(
            pt2
        ) == 2, f"Invalid pt length: expect 2, got {len(pt1)} and {len(pt2)}."
        return cls.from_list([*pt1, *pt2])

    def to_xywh_list(self) -> list[list[int]]:
        """return a xywh style list of this box.

        Returns:
            list[list[int]]: xywh list
        """
        x = (self.x1 + self.x2) // 2
        y = (self.y1 + self.y2) // 2
        w = (self.x2 - self.x1) // 2
        h = (self.y2 - self.y1) // 2
        return [[x, y], [w, h]]


@dataclasses.dataclass
class BasicInfo(object):
    loader: str
    video: Optional[str]
    mask: Optional[str]
    start_time: int
    end_time: int
    resolution: list[int]
    runtime_resolution: list[int]
    exp_time: float
    total_frames: int
    fps: float


@dataclasses.dataclass
class MDTarget(object):
    """Standard meteor detect target class.
    
    MDTarget describe a single result (a meteor, sprite), including its 

    Args:
        object (_type_): _description_
    """
    start_frame: int
    start_time: str
    end_time: str
    last_activate_frame: int
    last_activate_time: str
    duration: int
    speed: float
    dist: float
    fix_dist: float
    fix_speed: float
    fix_motion_duration: float
    fix_duration: float
    num_pts: int
    category: str
    pt1: list[int]
    pt2: list[int]
    center_point_list: list[list[int]]
    drct_loss: float
    score: float
    real_dist: float

    def to_simple_target(self):
        return SimpleTarget(pt1=self.pt1, pt2=self.pt2, preds=self.category)


@dataclasses.dataclass
class SingleMDRecord(object):
    """Meteor Detection single record.
    A record refers to a certain frame or a time clip,
    thus it contains list[MDTarget].

    Args:
        object (_type_): _description_
    """
    start_time: str
    end_time: str
    video_size: list[int]
    target: list[MDTarget]
    # TODO: 需要检查什么情况下会缺失这两个属性（理论上不应该缺失...）
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

    def to_video_data(self,
                      fps: Optional[float] = None,
                      video_size: Optional[list[int]] = None):
        """
        convert SingleMDRecord to VideoFrameData, for ClipToolkit using.

        Args:
            fps (float): video fps

        Raises:
            ValueError: if no num_frame is provided, 
            means it is not from video.

        Returns:
            VideoFrameData: converted video data.
        """

        return VideoFrameData(
            start_time=self.start_time,
            end_time=self.end_time,
            video_size=video_size,
            target_list=[x.to_simple_target() for x in self.target])

    def to_image_data(self):
        """
        SingleMDRecord should never converted to ImageFrameData.
        This function is for typing-check only.
        """
        raise ValueError("convert failed because img_filename is None.")


@dataclasses.dataclass
class SingleImgRecord(object):
    boxes: list[list[int]]
    preds: list[str]
    prob: list[str]
    img_filename: Optional[str] = None
    num_frame: Optional[int] = None

    def build_target_list(self):
        return [
            SimpleTarget(pt1=box[:2], pt2=box[2:], preds=pred)
            for (box, pred) in zip(self.boxes, self.preds)
        ]

    def frame2ts(self, frame: int, fps: float) -> str:
        return datetime.datetime.strftime(
            datetime.datetime.fromtimestamp(frame / fps,
                                            tz=datetime.timezone.utc),
            "%H:%M:%S.%f")[:-3]

    def to_video_data(self,
                      fps: Optional[float] = None,
                      video_size: Optional[list[int]] = None):
        """
        convert SingleImgRecord to VideoFrameData, for ClipToolkit using.

        Args:
            fps (float): video fps

        Raises:
            ValueError: if no num_frame is provided, 
            means it is not from video.

        Returns:
            VideoFrameData: converted video data.
        """
        if self.num_frame is None:
            raise ValueError("convert failed because num_frame is None.")
        assert fps is not None, f"fps should specified when converting {self.__class__.__name__}."
        assert len(self.boxes) == len(
            self.preds), (f"`preds` or `boxes` should have same length, "
                          f"got {len(self.boxes)} and {len(self.preds)}.")

        return VideoFrameData(start_time=self.frame2ts(self.num_frame, fps),
                              end_time=self.frame2ts(self.num_frame + 1, fps),
                              video_size=video_size,
                              target_list=self.build_target_list())

    def to_image_data(self):
        """
        convert SingleImgRecord to ImageFrameData, for ClipToolkit using.

        Raises:
            ValueError: if no img_filename is provided, 
            means it is not from image.
        Returns:
            ImageFrameData: converted image data.
        """
        if self.img_filename is None:
            raise ValueError("convert failed because img_filename is None.")
        return ImageFrameData(img_filename=self.img_filename,
                              target_list=self.build_target_list())


@dataclasses.dataclass
class MDRF(object):
    """Standard Meteor Detection Recording Format (for video).

    Args:
        object (_type_): _description_
    """
    version: str
    basic_info: BasicInfo
    config: dict[str, Any]
    type: str
    anno_size: list[int]
    results: Union[list[SingleMDRecord], list[SingleImgRecord]]


########### Model Config Dataclasses ################


@dataclasses.dataclass
class ModelCfg(object):
    name: str
    weight_path: str
    dtype: str
    nms: bool
    warmup: bool
    pos_thre: float
    nms_thre: float
    multiscale_pred: int
    multiscale_partition: int
    providers_key: Optional[str] = "default"


########### ClipToolkit Dataclasses ################


@dataclasses.dataclass
class ExportOption(object):
    exclude_category_list: list[str] = dataclasses.field(
        default_factory=lambda: [])
    jpg_quality: int = 95
    png_compressing: int = 3
    with_bbox: bool = False
    with_annotation: bool = False
    bbox_color: list[int] = dataclasses.field(
        default_factory=lambda: [255, 0, 0])
    bbox_thickness: int = 2
    video_encoder: str = "libx264"
    video_fmt: str = "yuv420p"


@dataclasses.dataclass
class ConnectParam(object):
    switch: bool
    ksize: int
    threshold: int


@dataclasses.dataclass
class SimpleDenoiseParam(object):
    ds_radius: int
    ds_threshold: int
    bi_d: int
    bi_sigma_color: int
    bi_sigma_space: int


@dataclasses.dataclass
class MFNRDenoiseParam(object):
    sigma_high: float
    sigma_low: float
    bg_fix_factor: float


@dataclasses.dataclass
class DenoiseOption(object):
    switch: bool
    highlight_preserve: float
    algorithm: str
    blur_ksize: int
    connect_lines: ConnectParam
    simple_param: SimpleDenoiseParam
    mfnr_param: MFNRDenoiseParam


@dataclasses.dataclass
class ClipCfg(object):
    loader: str
    wrapper: str
    writer: str
    image_denoise: DenoiseOption
    export: ExportOption


@dataclasses.dataclass
class ClipRequest(object):
    time: list[str]
    filename: Optional[str] = None
    target: Optional[list[dict[str, Any]]] = None

    def cvt_tgt(self):
        if self.target is None: return None
        return [
            from_dict(data_class=SimpleTarget, data=t) for t in self.target
        ]

    def to_video_data(self):
        return VideoFrameData(start_time=self.time[0],
                              end_time=self.time[1],
                              target_list=self.cvt_tgt(),
                              video_size=None,
                              saved_filename=self.filename)


@dataclasses.dataclass
class SimpleTarget(object):
    """
    Simple Target Class.
    
    Only contains necessary information for drawing and labelme annotation.
    """
    pt1: list[int]
    pt2: list[int]
    preds: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        bbox = Box.from_pts(self.pt1, self.pt2)
        return {
            "label": self.preds,
            "points": [[bbox.x1, bbox.y1], [bbox.x2, bbox.y2]],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None
        }


@dataclasses.dataclass
class ImageFrameData(object):
    img_filename: str
    target_list: list[SimpleTarget]
    img_size: Union[list[int], tuple[int, ...], None] = None
    saved_filename: Optional[str] = None

    def to_labelme(self) -> dict[str, Any]:
        w, h = None, None
        if self.img_size is not None and len(self.img_size) == 2:
            w, h = self.img_size
        if not self.saved_filename:
            raise FileNotFoundError(
                "Should not save labelme file without image filename.")
        return {
            "version": "5.5.0",
            "flags": {},
            "imagePath": self.saved_filename,
            "shapes": [target.to_json() for target in self.target_list],
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w
        }


@dataclasses.dataclass
class VideoFrameData(object):
    start_time: Optional[str]
    end_time: Optional[str]
    target_list: Optional[list[SimpleTarget]] = None
    video_size: Union[list[int], tuple[int, ...], None] = None
    saved_filename: Optional[str] = None

    def to_labelme(self) -> dict[str, Any]:
        w, h = None, None
        if self.video_size is not None and len(self.video_size) == 2:
            w, h = self.video_size
        if not self.saved_filename:
            raise FileNotFoundError(
                "Should not save labelme file without image filename.")
        return {
            "version":
            "5.5.0",
            "flags": {},
            "imagePath":
            self.saved_filename,
            "shapes":
            [target.to_json()
             for target in self.target_list] if self.target_list else None,
            "imageData":
            None,
            "imageHeight":
            h,
            "imageWidth":
            w
        }
