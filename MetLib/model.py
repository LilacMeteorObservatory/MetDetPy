from abc import ABCMeta, abstractmethod
from math import ceil
from os import path
import dataclasses
import json
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import onnxruntime as ort
import threading
from numpy.typing import DTypeLike, NDArray

from .imgproc import Size, normalize_size
from .metlog import BaseMetLog, get_default_logger
from .metstruct import (ClassifierCfg, ModelCfg, validate_num_threads,
                        validate_objectness_thre)
from .profiling import profiled
from .onnx_devices import discover_dml_adapters
from .utils import NUM_CLASS, STR2DTYPE, U8Mat, check_windows_dll, is_lfs_pointer, relative2abs_path, xywh2xyxy

ort.set_default_logger_severity(4)
logger = get_default_logger()

DEFAULT_STR = "default"
PARTITION_MIN_OVERLAP = 0.2
LOCK_TIMEOUT = 5.0
SUPPORTED_INPUT_COLOR_ORDERS = {"rgb", "bgr"}
DEFAULT_IMAGE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGE_STD = [0.229, 0.224, 0.225]
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



# ONNX Runtime 声明的输入类型 → numpy dtype
ORT_TYPE2DTYPE: dict[str, DTypeLike] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
}

# 分类器允许的显式精度
CLASSIFIER_INPUT_DTYPES: dict[str, DTypeLike] = {
    "float32": np.float32,
    "float16": np.float16,
}




def _tile_starts(length: int, size: int) -> list[int]:
    """Cover an axis with equal windows, anchoring both ends.

    Choose enough windows for at least PARTITION_MIN_OVERLAP, then distribute
    their starts. Computing window size and stride independently can leave an
    uncovered tail, even before either value is rounded to an integer.
    """
    if not 1 <= size <= length:
        raise ValueError("Tile size must be within the image axis.")
    if size == length:
        return [0]
    max_stride = max(1, int(size * (1 - PARTITION_MIN_OVERLAP)))
    intervals = ceil((length - size) / max_stride)
    return [i * (length - size) // intervals for i in range(intervals + 1)]


def _plan_prediction_tiles(height: int, width: int, hw_ratio: float,
                           tolerance: float, scales: int, partition: int):
    """Plan ratio adaptation and progressively smaller, overlapping views.

    Single-layer adaptation retains the full short axis and crops only the
    excess long axis to the model aspect ratio. Transposition is reserved for
    opposite image/model orientations. Resolution alone does not add views.
    Returned xyxy windows are in the possibly transposed input coordinates.
    """
    transpose = (abs(height / width - hw_ratio) > tolerance
                 and (height / width - 1) * (hw_ratio - 1) < 0)
    if transpose:
        height, width = width, height
    tile_h, tile_w = height, width
    if abs(height / width - hw_ratio) > tolerance:
        if height / width > hw_ratio:
            tile_h = max(1, min(height, round(width * hw_ratio)))
        else:
            tile_w = max(1, min(width, round(height / hw_ratio)))
    levels = []
    # A nominal partition=2 gives 2x2 overlapping views of a matching image.
    # Adapted images can need a different grid to preserve the tile ratio.
    shrink = partition - (partition - 1) * PARTITION_MIN_OVERLAP
    for scale in range(scales):
        if scale:
            tile_h = max(1, ceil(tile_h / shrink))
            tile_w = max(1, ceil(tile_w / shrink))
        windows = [(x, y, x + tile_w, y + tile_h)
                   for y in _tile_starts(height, tile_h)
                   for x in _tile_starts(width, tile_w)]
        levels.append(windows)
    return transpose, levels

def validate_provider_key(value: str) -> str:
    """Validate a backend alias with an optional nonnegative GPU index."""
    alias, separator, index = value.partition(":")
    if alias not in DEVICE_MAPPING:
        raise ValueError(f"Unknown provider: {value!r}.")
    if separator and (alias not in {"dml", "cuda"}
                      or not index.isascii() or not index.isdecimal()):
        raise ValueError("Device index must be a nonnegative integer, like: 'dml:0' or 'cuda:0'.")
    return value


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
                 logger: Optional[BaseMetLog] = None,
                 num_threads: int = 0) -> None:
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
    def forward(self, x: U8Mat) -> Optional[list[list[NDArray[np.float64]]]]:
        pass


class ONNXBackend(Backend):
    _device_locks: dict[str, threading.Lock] = {}
    _device_locks_guard = threading.Lock()

    @classmethod
    def _get_device_lock(cls, key: str):
        # Hold the registry guard only while locating/creating a device lock.
        with cls._device_locks_guard:
            if key not in cls._device_locks:
                conflict = (any(existing.startswith("dml:") for existing in cls._device_locks)
                            if key == "dml" else
                            key.startswith("dml:") and "dml" in cls._device_locks)
                if conflict:
                    raise ValueError(
                        "Cannot mix automatic DML (dml/default) with indexed DML "
                        "(dml:N) in one process. Use one DML selection mode consistently.")
                cls._device_locks[key] = threading.Lock()
            return cls._device_locks[key]

    def __init__(self,
                 weight_path: str,
                 dtype: Optional[DTypeLike],
                 warmup: bool,
                 providers_key: str,
                 logger: Optional[BaseMetLog] = None,
                 num_threads: int = 0) -> None:
        f"""Init a ONNXBackend that use onnxruntime as backend, supporting onnx format weight.
        Args:
            weight_path (str): /path/to/the/weight/file.
            dtype (Optional[np.dtype]): converted numpy data type; None means "follow the
                graph input type" (resolved from the session, e.g. fp16 weights).
            warmup (bool, optional): warmup to model before batch processing. Defaults to True.
            providers_key (str, optional): model provider. Defaults to None.
            logger (ThreadMetLog, optional): the stdout ThreadMetLog. Defaults to logger.
            num_threads (int): CPU intra-op threads including the calling thread.
                Defaults to 0 (ORT automatic selection); does not control GPU threads.
        """
        validate_num_threads(num_threads)
        self.weight_path = weight_path
        self.dtype = dtype
        self.logger = logger
        self._device_index = None
        indexed = bool(providers_key and ":" in providers_key)
        requested_provider = None
        # load model
        if indexed:
            validate_provider_key(providers_key)
            alias, index = providers_key.split(":")
            if alias not in {"dml", "cuda"}:
                raise ValueError(f"Unsupported ONNX provider: {providers_key!r}.")
            requested_provider = DEVICE_MAPPING[alias][0]
            if requested_provider not in ort.get_available_providers():
                raise ValueError(f"ONNX provider {requested_provider} is not installed.")
            self._device_index = int(index)
            self._device_key = f"{alias}:{self._device_index}"
            providers = [(requested_provider, {"device_id": str(self._device_index)}),
                         "CPUExecutionProvider"]
        else:
            if providers_key and providers_key not in DEVICE_MAPPING and self.logger:
                self.logger.warning(
                    f"Given provider {providers_key} is not supported. "
                    "Fall back to default provider.")
            providers = DEVICE_MAPPING.get(providers_key or DEFAULT_STR,
                                           DEVICE_MAPPING[DEFAULT_STR])
            # Default uses the configured provider family, without probing a Session.
            self._device_key = next(
                (alias for alias, choices in DEVICE_MAPPING.items()
                 if alias != DEFAULT_STR and choices[0] == providers[0]), providers[0])
        
        if is_lfs_pointer(self.weight_path):
            raise RuntimeError(
                f"Model weight file {self.weight_path} is a Git LFS pointer file. "
                "Please pull the actual model file using Git LFS."
            )
            
        self._lock = self._get_device_lock(self._device_key)
        session_kwargs = ({"enable_fallback": False}
                          if requested_provider is not None else {})
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = num_threads
        if any((provider[0] if isinstance(provider, tuple) else provider)
               == "DmlExecutionProvider" for provider in providers):
            # DirectML requires sequential execution and no memory pattern optimization.
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.enable_mem_pattern = False
        self.model_session = ort.InferenceSession(self.weight_path,
                                                  sess_options=session_options,
                                                  providers=providers,
                                                  **session_kwargs)
        if (requested_provider is not None
                and self.model_session.get_providers()[0] != requested_provider):
            raise RuntimeError(f"Failed to activate requested ONNX device {providers_key}.")
        self.shapes: list[list[int]] = [
            x.shape for x in self.model_session.get_inputs()
        ]
        self.names: list[str] = [
            x.name for x in self.model_session.get_inputs()
        ]

        self.single_input = True if len(self.shapes) == 1 else False

        if self.dtype is None:
            declared = (self.model_session.get_inputs()[0].type
                        if self.shapes else "")
            self.dtype = ORT_TYPE2DTYPE.get(declared, np.float32)

        # make sure batch=1
        # Warming up
        if warmup:
            # 动态维度（如分类器的 "batch"、动态 H/W）用 1 代替，只做一次形状预热
            with self._lock:
                _ = self.model_session.run(
                    [], {
                        name: np.zeros(
                            [dim if isinstance(dim, int) else 1 for dim in shape],
                            dtype=self.dtype)
                        for name, shape in zip(self.input_name, self.input_shape)
                    })

    @property
    def input_shape(self) -> list[list[int]]:
        return self.shapes

    @property
    def input_name(self) -> list[str]:
        return self.names

    @property
    def output_shape(self) -> list[list[int]]:
        return [x.shape for x in self.model_session.get_outputs()]

    @property
    def model_metadata(self) -> dict[str, str]:
        """权重文件自带的 `metadata_props`（如分类器的 `class_names`）。
        """
        return dict(self.model_session.get_modelmeta().custom_metadata_map)

    @property
    def device(self) -> str:
        provider = self.model_session.get_providers()[0]
        return f"{provider}:{self._device_index}" if self._device_index is not None else provider

    def forward(
        self, x: Union[U8Mat, list[U8Mat]]
    ) -> Optional[list[list[NDArray[np.float64]]]]:
        """ run inference session with given input.

        Args:
            x (Union[np.ndarray, list[np.ndarray]]): input tensor. Its type should varies
                according to the model. For model with multi-inputs, x should be a list of
                tensors; otherwise x should be a tensor.

        Returns:
            list: raw output, or None if the model lock could not be
            acquired within LOCK_TIMEOUT seconds.
        """
        assert len(self.input_name) > 0, "invalid input name cnt."
        acquired = self._lock.acquire(timeout=LOCK_TIMEOUT)
        if not acquired:
            if self.logger:
                self.logger.warning(
                    f"Failed to acquire lock for model {self.weight_path} within {LOCK_TIMEOUT} seconds.")
            return None
        try:
            if self.single_input:
                return self.model_session.run(None, {self.input_name[0]: x})

            return self.model_session.run(None, {
                name: tensor
                for name, tensor in zip(self.input_name, x)
            })
        finally:
            self._lock.release()


class YOLOModel(object):

    def __init__(self,
                 weight_path: str,
                 dtype: str,
                 nms: bool = False,
                 warmup: bool = True,
                 pos_thre: float = 0.25,
                 nms_thre: float = 0.45,
                 multiscale_pred: int = 1,
                 multiscale_partition: int = 2,
                 hw_tolerance: float = 0.2,
                 providers_key: Optional[str] = None,
                 input_color_order: str = "rgb",
                 logger: BaseMetLog = logger,
                 num_threads: int = 0,
                 objectness_thre: float = 0.25) -> None:
        r"""Init a YOLOModel that handles YOLO-like inputs and outputs.

        Args:
            weight_path (str): /path/to/the/weight/file.
            dtype (str): model dtype. Should be selected from float32, float16, int8.
            nms (bool, optional): whether to execute non-maximum suppression (NMS) for model outputs.
                If the model is not exported with nms, set to True. Defaults to False.
            warmup (bool, optional): warmup to model before batch processing. Defaults to True.
            pos_thre (float, optional): minimum output score
                ``sqrt(objectness * class_probability)``. Defaults to 0.25.
                NMS uses its square to compare raw joint scores.
            objectness_thre (float, optional): independent minimum objectness
                for raw candidates, applied before NMS. Defaults to 0.25;
                0 explicitly disables this gate.
            nms_thre (float, optional): NMS threshold when merging predictions. Defaults to 0.45.
            multiscale_pred (int, optional): the number of prediction scales, shoule be an integer>=0. 
                Different multiscale_pred scales performs as follows:
                1. When set to 0, there will be no extra transform before inference, just resize;
                2. When set to 1, adapt the aspect ratio with overlapping crops
                   at one scale; transpose only opposite image/model orientations;
                3. When number is larger than 1, the results will be predicted using multi-scale images
                    (similar to a feature pyramid).
                This can improve recall and precision, but may also increase the time cost. Defaults to 1.
            multiscale_partition (int, optional): The number of partitions for multi-scale images at each level.
                For example, partition=2 gives 2x2 overlapping views at the first
                sub-level of an aspect-matched image. Adapted inputs may need a
                different grid to preserve the crop ratio and cover all pixels.
                The default value is 2.
            hw_tolerance (float, optional): The max allowed scaling ratio. When running with multiscale mode,
                if diff height-width ratio is larger than excepted, image division will be applied.
            providers_key (str, optional): model provider. Defaults to None -> "default".
            input_color_order (str, optional): channel order expected by the
                model weights. Public ``forward`` inputs are always BGR.
                Supported values are ``"rgb"`` and ``"bgr"``. Defaults to
                ``"rgb"``.
            logger (ThreadMetLog, optional): the stdout ThreadMetLog. Defaults to logger.
            num_threads (int): Backend CPU inference threads, 0 for automatic.
                For ORT this controls intra-op parallelism, not inter-op or decoding.
        """
        self.weight_path = weight_path
        self.dtype = STR2DTYPE.get(dtype, np.float32)
        self.nms = nms
        validate_objectness_thre(objectness_thre)
        self.objectness_thre = objectness_thre
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

        if providers_key.startswith("dml:"):
            try:
                index = int(providers_key.split(":")[1])
                adapter = next((a for a in discover_dml_adapters() if a.index == index), None)
                if adapter is not None and adapter.duplicate_of is not None:
                    providers_key = f"dml:{adapter.duplicate_of}"
            except OSError as error:
                self.logger.warning(f"DML initialization identity query failed: {error}")

        # init model
        model_suffix = self.weight_path.split(".")[-1].lower()
        assert model_suffix in SUFFIX2BACKEND, f"Model arch not supported: only support {SUFFIX2BACKEND.keys()}, got {model_suffix}."
        self.BackendCls = SUFFIX2BACKEND[model_suffix]
        self.backend: Backend = self.BackendCls(self.weight_path,
                                                self.dtype,
                                                warmup,
                                                providers_key,
                                                logger=self.logger,
                                                num_threads=num_threads)
        self.logger.info(
            f"Sucessfully load {self.weight_path} on device= {self.backend.device} "
            f"with Warmup={warmup}, num_threads={num_threads} (0=auto)."
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
        else:
            self.resize = False

        # resize if necessary
        if self.resize:
            x = cv2.resize(x, (self.w, self.h), interpolation=cv2.INTER_CUBIC)

        # cvt h*w*c to b*c*h*w, and forward.
        x = (x.transpose(2, 0, 1))[None, ...]
        raw_output = self.backend.forward(x)
        if raw_output is None:
            # backend failed to acquire its lock in time; skip this input
            # instead of crashing, so the caller can keep running.
            self.logger.warning(
                f"Model {self.weight_path} skipped one forward call "
                "because the backend lock could not be acquired in time.")
            return (np.zeros((0, 4), dtype=np.int_),
                    np.zeros((0, NUM_CLASS), dtype=np.float64))
        results = raw_output[0][0]

        # Gate raw candidates before class scoring or suppression. A high class
        # score must not let a low-objectness box suppress an eligible target.
        if self.objectness_thre > 0:
            results = results[results[:, 4] > self.objectness_thre]

        # YOLO output: [0:4] is center-based xywh, 4 is objectness and [5:]
        # contains per-class scores.
        objectness = results[:, 4]
        joint_scores: NDArray[np.float64] = np.einsum(
            "ab,a->ab", results[:, 5:], objectness)
        if self.nms:
            boxes_tlwh = _cxcywh_to_tlwh(results[:, :4])
            selected = _class_aware_nms(boxes_tlwh, joint_scores,
                                        self.pos_thre ** 2, self.nms_thre)
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

    @profiled("model.forward")
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

        transpose_flag, levels = _plan_prediction_tiles(
            h, w, self.hw_ratio, self.hw_tolerance, self.multiscale_pred,
            self.multiscale_partition)
        if transpose_flag:
            x = np.transpose(x, (1, 0, 2))
        tot_partition_num = sum(len(windows) for windows in levels)
        self.logger.debug(
            f"Forward with total partition: {tot_partition_num}; image transpose: {transpose_flag}"
        )

        # Each level covers both image edges; windows keep the base crop ratio.
        result_pos: list[NDArray[np.int_]] = []
        result_joint_scores: list[NDArray[np.float64]] = []
        try:
            for windows in levels:
                for x1, y1, x2, y2 in windows:
                    clip_pos, clip_joint_scores = self._forward(x[y1:y2, x1:x2])
                    clip_pos[:, [0, 2]] += x1
                    clip_pos[:, [1, 3]] += y1
                    result_pos.append(clip_pos)
                    result_joint_scores.append(clip_joint_scores)
        except Exception as e:
            # 异常跳过
            self.logger.error(
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
                                    self.pos_thre ** 2, self.nms_thre)
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

@dataclasses.dataclass
class ClassifierSpec:
    """ROI 分类器的初始化契约

    Args:
        input_size (Size): 输入尺寸（``width`` / ``height`` 显式命名）。方形只是本仓库
            自带权重的选择，框架本身接受任意固定尺寸。
        color (str): 输入色彩顺序，"rgb" 或 "bgr"。
        mean (list[float]): 逐通道归一化均值（RGB 顺序）。
        std (list[float]): 逐通道归一化标准差（RGB 顺序）。
        class_names (list[str]): 输出索引 → 类别名。
        background_index (Optional[int]): 背景类的输出索引；None 表示这份
            权重没有背景类，闸门因此无法拒绝任何候选（等同关闭）。
    """

    input_size: Size
    color: str
    mean: list[float]
    std: list[float]
    class_names: list[str]
    background_index: Optional[int] = None


def make_classifier_spec(input_size: Union[int, Sequence[int], Size],
                         color: str = "rgb",
                         mean: Optional[list[float]] = None,
                         std: Optional[list[float]] = None,
                         class_names: Optional[list[str]] = None,
                         background_id: Optional[int] = None,
                         background_name: Optional[str] = "BACKGROUND"
                         ) -> ClassifierSpec:
    """把配置与权重里读到的信息校验成 `ClassifierSpec`（不读文件，便于单测）。

    Args:
        input_size (Union[int, Sequence[int], Size]): 输入尺寸。
            标量=正方形，``(w, h)``=长方形。
        color (str): 输入色彩顺序。
        mean (Optional[list[float]]): 归一化均值；None 用 ImageNet 默认值。
        std (Optional[list[float]]): 归一化标准差；None 用 ImageNet 默认值。
        class_names (Optional[list[str]]): 类别名（配置优先，其次权重内置的 class_names）。
        background_id (Optional[int]): 背景索引；给了就不再按名字解析。
        background_name (Optional[str]): 背景类名；None 表示这份权重没有背景类。

    Returns:
        ClassifierSpec: 校验后的初始化契约。

    Raises:
        ValueError: 任一字段缺失、越界或与类别表不一致。
    """
    input_size = (input_size if isinstance(input_size, Size)
                  else normalize_size(input_size))
    color = str(color).lower()
    if color not in SUPPORTED_INPUT_COLOR_ORDERS:
        raise ValueError(
            f"Unsupported classifier input color order: {color!r} "
            f"(expected one of {sorted(SUPPORTED_INPUT_COLOR_ORDERS)}).")
    mean = [float(x) for x in (mean if mean is not None else DEFAULT_IMAGE_MEAN)]
    std = [float(x) for x in (std if std is not None else DEFAULT_IMAGE_STD)]
    for name, values in (("mean", mean), ("std", std)):
        if len(values) != 3:
            raise ValueError(
                f"Classifier {name} must have 3 channel values, got {values!r}.")
        if name == "std" and any(value <= 0 for value in values):
            raise ValueError(
                f"Classifier std must be positive, got {values!r}.")
    if not class_names:
        raise ValueError(
            "No class names for the ROI classifier: pass `classes` in "
            "`recheck_cfg.gate`, or use a weight file with embedded "
            "`class_names` metadata.")
    names = [str(name) for name in class_names]
    if len(set(names)) != len(names):
        raise ValueError(f"Classifier class names are not unique: {names}.")

    background_index = background_id
    if background_index is None and background_name:
        if background_name not in names:
            raise ValueError(
                f"Background class {background_name!r} is not in the classifier "
                f"class table {names}; set `background_name` to the right label "
                "or null when the model has no background class.")
        background_index = names.index(background_name)
    if background_index is not None:
        if not isinstance(background_index, int):
            raise ValueError(
                f"Classifier background_id must be an integer, got "
                f"{background_index!r}.")
        if not 0 <= background_index < len(names):
            raise ValueError(
                f"Classifier background_id {background_index} is out of range "
                f"for {len(names)} classes.")
    return ClassifierSpec(input_size=input_size,
                          color=color,
                          mean=mean,
                          std=std,
                          class_names=names,
                          background_index=background_index)


class ROIClassifierModel(object):
    """ROI 分类器：输入已裁好的 ROI，输出各类概率。

    与 `YOLOModel` 的区别：不产生检测框，不做多尺度切片，也不做 NMS；输入是单张
    （或一批）ROI。预处理按 `ClassifierCfg` 的初始化契约执行：BGR→RGB、双线性
    抗锯齿缩放到 `input_size`、`/255`、逐通道 mean/std、HWC→CHW、按权重精度 cast。

    契约的来源：
      1. 输入尺寸、张量名、类别名与精度**来自权重文件**（固定的 (w, h) 输入形状、
         会话声明的张量名输入类型、`metadata_props.class_names`）；
      2. `ClassifierCfg` 只配置那些自由度的项：`input_color_order`、`mean`/`std`、
         `classes`（覆盖权重内置类别表）、`background_name`/`background_id`、`dtype`
         （覆盖权重精度）。

    Args:
        cfg (ClassifierCfg): 闸门/分类器配置（`switch` 由调用方判断）。
        logger (Optional[BaseMetLog]): 日志。
    """

    def __init__(self,
                 cfg: ClassifierCfg,
                 logger: Optional[BaseMetLog] = None) -> None:
        if cfg is None:
            raise ValueError("ROIClassifierModel needs a ClassifierCfg.")
        if not cfg.weight_path:
            raise ValueError(
                "recheck_cfg.gate.weight_path is empty; point it at an onnx "
                "weight (the shipped default is "
                "./weights/mobilenet_v3_large_v2_alpha_fp16.onnx).")
        self.logger = logger or get_default_logger()
        self.weight_path = (cfg.weight_path if path.isabs(cfg.weight_path) else
                            relative2abs_path(cfg.weight_path))
        if is_lfs_pointer(self.weight_path):
            raise RuntimeError(
                f"Model weight file {self.weight_path} is a Git LFS pointer file.")
        validate_num_threads(cfg.num_threads)
        if cfg.dtype == "auto":
            input_dtype = None
        elif cfg.dtype in CLASSIFIER_INPUT_DTYPES:
            input_dtype = CLASSIFIER_INPUT_DTYPES[cfg.dtype]
        else:
            raise ValueError(
                f"Unsupported classifier dtype {cfg.dtype!r} (expected 'auto' "
                f"or one of {sorted(CLASSIFIER_INPUT_DTYPES)}).")
        self.backend = ONNXBackend(self.weight_path,
                                   input_dtype,
                                   cfg.warmup,
                                   providers_key=cfg.providers_key,
                                   logger=self.logger,
                                   num_threads=cfg.num_threads)
        self.input_dtype = self.backend.dtype

        shapes = self.backend.input_shape
        if len(shapes) != 1:
            raise ValueError(
                f"ROI classifier expects a single input, got {len(shapes)}.")
        shape = shapes[0]
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(
                f"ROI classifier input must be NCHW with 3 channels, got {shape}.")
        height, width = shape[2], shape[3]
        # 输入尺寸只由权重决定
        if not (isinstance(height, int) and isinstance(width, int)):
            raise ValueError(
                "ROI classifier weight must declare a fixed input height/width, "
                f"got {shape}. Re-export the model with static spatial dims.")
        input_size = Size(width=width, height=height)

        # 类别名：配置优先，其次用权重自带的 class_names。
        class_names = cfg.classes
        if not class_names:
            embedded = self.backend.model_metadata.get("class_names")
            if embedded:
                class_names = json.loads(embedded)
        self.spec = make_classifier_spec(
            input_size=input_size,
            color=cfg.input_color_order,
            mean=cfg.mean,
            std=cfg.std,
            class_names=class_names,
            background_id=cfg.background_id,
            background_name=cfg.background_name)

        output_shape = self.backend.output_shape[0] if self.backend.output_shape else []
        num_output = output_shape[-1] if output_shape else None
        if isinstance(num_output, int) and num_output != len(self.spec.class_names):
            raise ValueError(
                f"Classifier outputs {num_output} classes but the class table has "
                f"{len(self.spec.class_names)}: {self.spec.class_names}.")

    @property
    def class_names(self) -> list[str]:
        return self.spec.class_names

    @property
    def background_index(self) -> Optional[int]:
        return self.spec.background_index

    @property
    def input_size(self) -> Size:
        """模型输入尺寸，即"画布"尺寸（``width`` / ``height`` 显式命名）。"""
        return self.spec.input_size

    def preprocess(self, roi: U8Mat) -> NDArray[np.float32]:
        """把一张已裁好的 ROI 处理成模型输入（N=1）。

        Args:
            roi (U8Mat): HWC、3 通道、BGR（OpenCV 约定）的 ROI。

        Returns:
            NDArray[np.float32]: 形状 [1, 3, S, S] 的 float32 张量（未按精度 cast）。
        """
        assert roi.ndim == 3 and roi.shape[2] == 3, \
            f"ROI must be HWC with 3 channels, got {roi.shape}."
        size = self.spec.input_size
        if roi.shape[0] < size.height or roi.shape[1] < size.width:
            # 任一轴需要放大就用 LINEAR（放大时 AREA 会退化成近似最近邻）。
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_AREA
        resized = cv2.resize(roi, size.as_wh(), interpolation=interpolation)
        pixels = resized[:, :, ::-1] if self.spec.color == "rgb" else resized
        x = pixels.astype(np.float32) / 255.0
        mean = np.asarray(self.spec.mean, dtype=np.float32)
        std = np.asarray(self.spec.std, dtype=np.float32)
        x = (x - mean) / std
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)

    def forward(self, roi: Union[U8Mat, list[U8Mat]]) -> NDArray[np.float64]:
        """对一张或一批 ROI 推理，返回 softmax 后的类别概率。

        Args:
            roi (Union[U8Mat, list[U8Mat]]): 单张 ROI 或 ROI 列表。

        Returns:
            NDArray[np.float64]: 形状 [N, num_class] 的概率。
        """
        rois = [roi] if isinstance(roi, np.ndarray) else list(roi)
        if not rois:
            raise ValueError("ROIClassifierModel.forward got no ROI.")
        batch = np.concatenate([self.preprocess(x) for x in rois], axis=0)
        batch = batch.astype(self.input_dtype)
        raw = self.backend.forward(batch)
        if raw is None:
            return np.zeros((0, len(self.class_names)), dtype=np.float64)
        logits = np.asarray(raw[0], dtype=np.float64)
        if logits.ndim == 1:
            logits = logits[None, :]
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, roi: U8Mat) -> tuple[str, float]:
        """单张 ROI 的分类结果。

        Args:
            roi (U8Mat): 已裁好的 BGR ROI。

        Returns:
            tuple[str, float]: (类别名, 概率)。
        """
        probs = self.forward(roi)[0]
        index = int(np.argmax(probs))
        return self.class_names[index], float(probs[index])

    def background_probability(self, roi: U8Mat) -> float:
        """背景类的概率；权重没有背景类时返回 0。"""
        index = self.background_index
        if index is None or index >= len(self.class_names):
            return 0.0
        return float(self.forward(roi)[0][index])


def init_classifier(cfg: ClassifierCfg,
                    logger: BaseMetLog) -> Optional[ROIClassifierModel]:
    """按 `ClassifierCfg` 构造 ROI 分类器；未启用时返回 None。

    Args:
        cfg (ClassifierCfg): 闸门/分类器配置。
        logger (BaseMetLog): 日志。

    Returns:
        Optional[ROIClassifierModel]: 启用时返回模型实例。
    """
    if cfg is None or not cfg.switch:
        return None
    model = ROIClassifierModel(cfg, logger)
    background = ("none" if model.background_index is None else
                  f"{model.background_index}:"
                  f"{model.class_names[model.background_index]}")
    operating_point = (f"p_background_thre={cfg.p_background_thre}"
                       if cfg.decision == "threshold" else
                       "p_background_thre unused")
    crop = ("legacy floor"
            if cfg.target_long_side is None else
            f"target_long_side={cfg.target_long_side}"
            f" (floor {cfg.adaptive_side_floor})")
    model.logger.info(
        f"ROI gate ready: {model.weight_path} on {model.backend.device}; "
        f"input {model.spec.input_size.width}x{model.spec.input_size.height} "
        f"{model.spec.color.upper()}, dtype={np.dtype(model.input_dtype).name}, "
        f"classes={len(model.class_names)}, background={background}, "
        f"decision={cfg.decision}, {operating_point}, crop={crop}.")
    return model


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
                 num_threads=cfg.num_threads,
                 objectness_thre=cfg.objectness_thre,
                 logger=logger)
