from abc import ABCMeta, abstractmethod
from os import path
from typing import Optional, Union

import cv2
import numpy as np
import onnxruntime as ort
import threading
from numpy.typing import DTypeLike, NDArray

from .metlog import BaseMetLog, get_default_logger
from .metstruct import ModelCfg
from .utils import NUM_CLASS, STR2DTYPE, U8Mat, check_windows_dll, is_lfs_pointer, relative2abs_path, xywh2xyxy

ort.set_default_logger_severity(4)
logger = get_default_logger()

DEFAULT_STR = "default"
PARTITION_MIN_OVERLAP = 0.2
LOCK_TIMEOUT = 5.0
SUPPORTED_INPUT_COLOR_ORDERS = {"rgb", "bgr"}

DEVICE_MAPPING: dict[str, list[str]] = {
    "cpu": ["CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider"],
    "cuda": ["CUDAExecutionProvider"],
    DEFAULT_STR: ort.get_available_providers(),
    "coreml": ["CoreMLExecutionProvider"]
}

AVAILABLE_DEVICE_ALIAS = [
    alias for (alias, pvd_list) in DEVICE_MAPPING.items()
    if pvd_list[0] in ort.get_available_providers()
]

WINDOWS_DLL_CHK_LIST = [
        "vcruntime140_1.dll",
        "vcruntime140.dll",
        "msvcp140.dll",
        "ucrtbase.dll"
    ]


def _cxcywh_to_tlwh(boxes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert YOLO center-based boxes to top-left ``(x, y, w, h)``."""
    converted = np.asarray(boxes, dtype=np.float32).copy()
    converted[:, 0] -= converted[:, 2] / 2
    converted[:, 1] -= converted[:, 3] / 2
    return converted


def _xyxy_to_tlwh(boxes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert corner-based boxes to top-left ``(x, y, w, h)``."""
    converted = np.asarray(boxes, dtype=np.float32).copy()
    converted[:, 2] -= converted[:, 0]
    converted[:, 3] -= converted[:, 1]
    return converted


def _class_aware_nms(boxes_tlwh: NDArray[np.float64],
                     class_scores: NDArray[np.float64],
                     score_threshold: float,
                     nms_threshold: float) -> NDArray[np.intp]:
    """Run class-aware OpenCV NMS on top-left xywh boxes."""
    if len(boxes_tlwh) != len(class_scores):
        raise ValueError("boxes_tlwh and class_scores must have the same length")
    if len(boxes_tlwh) == 0:
        return np.empty(0, dtype=np.intp)
    if class_scores.ndim != 2 or class_scores.shape[1] == 0:
        raise ValueError("class_scores must have shape (n, num_classes)")

    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)
    selected = cv2.dnn.NMSBoxesBatched(
        bboxes=boxes_tlwh.tolist(),
        scores=scores.tolist(),
        class_ids=class_ids.tolist(),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold)
    selected = np.asarray(selected, dtype=np.intp).reshape(-1)
    return selected[np.argsort(scores[selected])[::-1]]


class Backend(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self,
                 weight_path: str,
                 dtype: DTypeLike,
                 warmup: bool,
                 providers_key: Optional[str] = None,
                 logger: Optional[BaseMetLog] = None) -> None:
        pass

    @property
    def input_shape(self) -> list[list[int]]:
        pass

    @property
    @abstractmethod
    def input_name(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        pass

    @abstractmethod
    def forward(self, x: U8Mat) -> list[list[NDArray[np.float64]]]:
        pass


class ONNXBackend(Backend):
    _global_lock = threading.Lock()
    def __init__(self,
                 weight_path: str,
                 dtype: DTypeLike,
                 warmup: bool,
                 providers_key: Optional[str] = None,
                 logger: Optional[BaseMetLog] = None) -> None:
        f"""Init a ONNXBackend that use onnxruntime as backend, supporting onnx format weight.
        Args:
            weight_path (str): /path/to/the/weight/file.
            dtype (np.dtype): converted numpy data type.
            warmup (bool, optional): warmup to model before batch processing. Defaults to True.
            providers_key (str, optional): model provider. Defaults to None.
            logger (ThreadMetLog, optional): the stdout ThreadMetLog. Defaults to logger.
        """
        self.weight_path = weight_path
        self.dtype = dtype
        # load model
        if providers_key and (not providers_key in DEVICE_MAPPING) and logger:
            logger.warning(
                f"Gicen provider {providers_key} is not supported." +
                "Fall back to default provider.")
        if not providers_key:
            providers = DEVICE_MAPPING[DEFAULT_STR]
        else:
            providers = DEVICE_MAPPING.get(providers_key,
                                           DEVICE_MAPPING[DEFAULT_STR])
        
        if is_lfs_pointer(self.weight_path):
            raise RuntimeError(
                f"Model weight file {self.weight_path} is a Git LFS pointer file. "
                "Please pull the actual model file using Git LFS."
            )
            
        self.model_session = ort.InferenceSession(self.weight_path,
                                                  providers=providers)
        self.shapes: list[list[int]] = [
            x.shape for x in self.model_session.get_inputs()
        ]
        self.names: list[str] = [
            x.name for x in self.model_session.get_inputs()
        ]

        self.single_input = True if len(self.shapes) == 1 else False

        # make sure batch=1
        # Warming up
        if warmup:
            # TODO: dynamic 模型的返回的值为 ['images'] [['batch', 3, 'height', 'width']]
            # 无法适配当前backend模型
            _ = self.model_session.run(
                [], {
                    name: np.zeros(shape, dtype=self.dtype)
                    for name, shape in zip(self.input_name, self.input_shape)
                })

    @property
    def input_shape(self) -> list[list[int]]:
        return self.shapes

    @property
    def input_name(self) -> list[str]:
        return self.names

    @property
    def device(self) -> str:
        return self.model_session.get_providers()[0]

    def forward(
            self, x: Union[U8Mat,
                           list[U8Mat]]) -> list[list[NDArray[np.float64]]]:
        """ run inference session with given input.

        Args:
            x (Union[np.ndarray, list[np.ndarray]]): input tensor. Its type should varies
                according to the model. For model with multi-inputs, x should be a list of
                tensors; otherwise x should be a tensor.

        Returns:
            list: raw output.
        """
        assert len(self.input_name) > 0, "invalid input name cnt."
        self._global_lock.acquire(timeout=LOCK_TIMEOUT)
        try:
            if self.single_input:
                return self.model_session.run(None, {self.input_name[0]: x})

            return self.model_session.run(None, {
                name: tensor
                for name, tensor in zip(self.input_name, x)
            })
        finally:
            self._global_lock.release()


class YOLOModel(object):

    def __init__(self,
                 weight_path: str,
                 dtype: str,
                 nms: bool = False,
                 warmup: bool = True,
                 pos_thre: float = 0.10,
                 nms_thre: float = 0.45,
                 multiscale_pred: int = 1,
                 multiscale_partition: int = 2,
                 hw_tolerance: float = 0.2,
                 providers_key: Optional[str] = None,
                 input_color_order: str = "rgb",
                 logger: BaseMetLog = logger) -> None:
        r"""Init a YOLOModel that handles YOLO-like inputs and outputs.

        Args:
            weight_path (str): /path/to/the/weight/file.
            dtype (str): model dtype. Should be selected from float32, float16, int8.
            nms (bool, optional): whether to execute non-maximum suppression (NMS) for model outputs.
                If the model is not exported with nms, set to True. Defaults to False.
            warmup (bool, optional): warmup to model before batch processing. Defaults to True.
            pos_thre (float, optional): minimum raw ``objectness * class_probability``
                score used by NMS to keep positive samples. Defaults to 0.10.
            nms_thre (float, optional): NMS threshold when merging predictions. Defaults to 0.45.
            multiscale_pred (int, optional): the number of prediction scales, shoule be an integer>=0. 
                Different multiscale_pred scales performs as follows:
                1. When set to 0, there will be no extra transform before inference, just resize;
                2. When set to 1, there will be an optional basic transform and split on the full image;
                3. When number is larger than 1, the results will be predicted using multi-scale images
                    (similar to a feature pyramid).
                This can improve recall and precision, but may also increase the time cost. Defaults to 1.
            multiscale_partition (int, optional): The number of partitions for multi-scale images at each level.
                For example, at the first sub-level, if the multiscale_partition is set to 2, the image will 
                be divided into $\(2^{1} \times 2^{1} = 2 \times 2\)$ sub-images and sent to the model. 
                The default value is 2.
            hw_tolerance (float, optional): The max allowed scaling ratio. When running with multiscale mode,
                if diff height-width ratio is larger than excepted, image division will be applied.
            providers_key (str, optional): model provider. Defaults to None -> "default".
            input_color_order (str, optional): channel order expected by the
                model weights. Public ``forward`` inputs are always BGR.
                Supported values are ``"rgb"`` and ``"bgr"``. Defaults to
                ``"rgb"``.
            logger (ThreadMetLog, optional): the stdout ThreadMetLog. Defaults to logger.
        """
        self.weight_path = weight_path
        self.dtype = STR2DTYPE.get(dtype, np.float32)
        self.nms = nms
        self.pos_thre = pos_thre
        self.nms_thre = nms_thre
        self.logger = logger
        self.unwarning = True
        self.resize = False
        self.multiscale_pred = multiscale_pred
        self.multiscale_partition = multiscale_partition
        self.hw_tolerance = hw_tolerance
        if not isinstance(input_color_order, str):
            raise ValueError(
                f"Model input color order must be a string, got "
                f"{type(input_color_order).__name__}.")
        self.input_color_order = input_color_order.lower()
        if self.input_color_order not in SUPPORTED_INPUT_COLOR_ORDERS:
            raise ValueError(
                f"Unsupported model input color order: {input_color_order!r}. "
                f"Expected one of {sorted(SUPPORTED_INPUT_COLOR_ORDERS)}.")
        if providers_key is None:
            providers_key = DEFAULT_STR

        # init model
        model_suffix = self.weight_path.split(".")[-1].lower()
        assert model_suffix in SUFFIX2BACKEND, f"Model arch not supported: only support {SUFFIX2BACKEND.keys()}, got {model_suffix}."
        self.BackendCls = SUFFIX2BACKEND[model_suffix]
        self.backend: Backend = self.BackendCls(self.weight_path,
                                                self.dtype,
                                                warmup,
                                                providers_key,
                                                logger=self.logger)
        self.logger.info(
            f"Sucessfully load {self.weight_path} on device= {self.backend.device} with Warmup={warmup}."
        )

        # for yolo only first argument is working.
        self.b, self.c, self.h, self.w = self.backend.input_shape[0]
        self.hw_ratio = self.h / self.w
        self.scale_w, self.scale_h = 1, 1

    def _forward(self, x: U8Mat):
        """simple forward function with rescale.

        Args:
            x (np.ndarray): input image. should be 3-channel.

        Returns:
            tuple: a pair of list be like (pred_pos, pred_cls).
        """
        h, w, c = x.shape
        assert c == self.c, "num_channel must match."

        if (h != self.h or w != self.w):
            self.resize = True
            self.scale_h, self.scale_w = h / self.h, w / self.w
            # 仅在第一次运行不匹配时抛出Warning
            if self.unwarning:
                self.logger.warning(
                    f"Model input shape ({self.h}x{self.w}) is "
                    f"not strictly matched with config ({h}x{w}). "
                    f"Extra resize is applied to avoid error (which may increase time cost.)"
                )
                self.unwarning = False

        # resize if necessary
        if self.resize:
            x = cv2.resize(x, (self.w, self.h), interpolation=cv2.INTER_CUBIC)

        # cvt h*w*c to b*c*h*w, and forward.
        x = (x.transpose(2, 0, 1))[None, ...]
        results = self.backend.forward(x)[0][0]

        # YOLO output: [0:4] is center-based xywh, 4 is objectness and [5:]
        # contains per-class scores.
        objectness = results[:, 4]
        joint_scores: NDArray[np.float64] = np.einsum(
            "ab,a->ab", results[:, 5:], objectness)
        if self.nms:
            boxes_tlwh = _cxcywh_to_tlwh(results[:, :4])
            selected = _class_aware_nms(boxes_tlwh, joint_scores,
                                        self.pos_thre, self.nms_thre)
            results = results[selected]
            joint_scores = joint_scores[selected]

        # The rest of the pipeline uses corner-based boxes for scaling,
        # per-tile offsets and final output.
        xywh2xyxy(results[:, :4], inplace=True)

        # resize back if necessary
        if self.resize:
            results[:, 0] *= self.scale_w
            results[:, 2] *= self.scale_w
            results[:, 1] *= self.scale_h
            results[:, 3] *= self.scale_h
        # 整数化坐标，类别输出概率矩阵
        result_pos: NDArray[np.int_] = np.array(results[:, :4], dtype=int)
        return result_pos, joint_scores

    def forward(self, x: U8Mat):
        """forward function that supports multiscale inference.
        
        Results will be recombined.

        Args:
            x (_type_): _description_

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        assert isinstance(x, np.ndarray) and len(
            x.shape) == 3, "input x must be a 3-dim array!"
        h, w, c = x.shape
        assert h > 0 and w > 0 and c == self.c, f"input array shapemust be valid, got {x.shape}."
        assert x.dtype == np.uint8, f"input array must be uint8, got {x.dtype}."

        # Public image inputs follow the project's OpenCV BGR convention.
        # Convert once before any multi-scale partitioning when the model was
        # trained with RGB inputs.
        if self.input_color_order == "rgb":
            x = x[..., ::-1]
        elif self.input_color_order != "bgr":
            raise ValueError(
                f"Unsupported model input color order: {self.input_color_order!r}.")

        # 预处理：转换为dtype并归一化[0,1]范围。
        # TODO: 未兼容INT8场合。
        x = np.ascontiguousarray(x, dtype=self.dtype) / 255

        if self.multiscale_pred == 0:
            result_pos, joint_scores = self._forward(x)
            return result_pos, np.sqrt(joint_scores)

        transpose_flag = False
        input_hw_ratio = h / w
        h_rep, w_rep = 1, 1
        if abs(self.hw_ratio - input_hw_ratio) > self.hw_tolerance:
            # 需要对第一级做分割，判断是否需要旋转底图：
            # 1. 模型和原图同向
            # 2. 长短边异向，但旋转后符合直接匹配
            if ((input_hw_ratio - 1) * (self.hw_ratio - 1)) > 0 or abs(
                    self.hw_ratio - 1 / input_hw_ratio) < self.hw_tolerance:
                transpose_flag = True
                x = np.transpose(x, (1, 0, 2))
                input_hw_ratio = 1 / input_hw_ratio
                h, w = w, h
            # TODO: 这段逻辑比较绕...归纳出统一的格式
            if h > w:
                h_rep = np.ceil(h * self.w / (self.h * w)).astype(int)
            else:
                w_rep = np.ceil(w * self.h / (h * self.w)).astype(int)
        n = self.multiscale_partition**2
        tot_partition_num = h_rep * w_rep * (n**(self.multiscale_pred) -
                                             1) // (n - 1)
        self.logger.debug(
            f"Forward with total partition: {tot_partition_num}; image transpose: {transpose_flag}"
        )

        # 根据 rep 情况 分检测用 patch_index_list
        result_pos: list[NDArray[np.int_]] = []
        result_joint_scores: list[NDArray[np.float64]] = []
        try:
            for scale in range(self.multiscale_pred):
                if scale > 0:
                    h_rep *= self.multiscale_partition
                    w_rep *= self.multiscale_partition
                tot_h_rep = (h_rep - 1) * PARTITION_MIN_OVERLAP
                tot_w_rep = (w_rep - 1) * PARTITION_MIN_OVERLAP
                h_size = int(h // (h_rep - tot_h_rep))
                w_size = int(w // (w_rep - tot_w_rep))
                h_stride = int(h // (h_rep + tot_h_rep))
                w_stride = int(w // (w_rep + tot_w_rep))

                for i in range(h_rep):
                    for j in range(w_rep):
                        clip_img = x[i * h_stride:i * h_stride + h_size,
                                     j * w_stride:j * w_stride + w_size]
                        clip_pos, clip_joint_scores = self._forward(clip_img)
                        clip_pos[:, 1] += i * h_stride
                        clip_pos[:, 3] += i * h_stride
                        clip_pos[:, 0] += j * w_stride
                        clip_pos[:, 2] += j * w_stride
                        result_pos.append(clip_pos)
                        result_joint_scores.append(clip_joint_scores)
        except Exception as e:
            # 异常跳过
            logger.error(
                f"Exception {e.__repr__()} encountered with calling {self.__class__.__name__}. "
                f"Results of this frame could be lost...")
            if len(result_pos) == 0 or len(result_joint_scores) == 0:
                return np.zeros((0, 4), dtype=np.int_), np.zeros(
                    (0, NUM_CLASS), dtype=np.float64)
            return (np.concatenate(result_pos, axis=0),
                    np.sqrt(np.concatenate(result_joint_scores, axis=0)))
        concat_result_pos = np.concatenate(result_pos, axis=0)
        concat_joint_scores = np.concatenate(result_joint_scores, axis=0)

        # Merge tiles with the same coordinate, score and class semantics used
        # by the per-tile NMS. Reuse the configurable IoU threshold rather than
        # a hard-coded aggressive threshold.
        boxes_tlwh = _xyxy_to_tlwh(concat_result_pos)
        selected = _class_aware_nms(boxes_tlwh, concat_joint_scores,
                                    self.pos_thre, self.nms_thre)
        concat_result_pos = concat_result_pos[selected]
        concat_joint_scores = concat_joint_scores[selected]

        # 输出前将结果转置回来
        if transpose_flag:
            concat_result_pos = concat_result_pos[:, [1, 0, 3, 2]]

        # prob以修正分数，得分会很低，因此使用sqrt()的得分修正公式。
        # TODO: 通过优化模型取缔这个tricky的设置。
        return concat_result_pos, np.sqrt(concat_joint_scores)

check_windows_dll(WINDOWS_DLL_CHK_LIST)
available_models = {cls.__name__: cls for cls in [YOLOModel]}
SUFFIX2BACKEND = {"onnx": ONNXBackend}


def init_model(cfg: ModelCfg, logger: BaseMetLog):
    """ 兼容现有 Easydict 数据类型的模型初始化实现。
    NOTE: 因为 Model 可能会被其他子模块使用，因此单独实现了初始化模块。

    Args:
        cfg (ModelCfg): _description_
        logger (BaseMetLog): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    if not cfg.name in available_models:
        raise Exception(f"No model named {cfg.name}.")
    Model = available_models[cfg.name]
    weight_path = relative2abs_path(cfg.weight_path) if not path.isabs(cfg.weight_path) else cfg.weight_path
    return Model(weight_path=weight_path,
                 dtype=cfg.dtype,
                 nms=cfg.nms,
                 warmup=cfg.warmup,
                 pos_thre=cfg.pos_thre,
                 nms_thre=cfg.nms_thre,
                 multiscale_pred=cfg.multiscale_pred,
                 multiscale_partition=cfg.multiscale_partition,
                 providers_key=cfg.providers_key,
                 input_color_order=cfg.input_color_order,
                 logger=logger)
