from abc import ABCMeta, abstractmethod
from typing import Optional, Union
import cv2
import numpy as np
import onnxruntime as ort

from abc import abstractmethod, ABCMeta
from .metlog import get_default_logger
from .utils import xywh2xyxy, STR2DTYPE

ort.set_default_logger_severity(3)
logger = get_default_logger()

DEVICE_MAPPING = {
    "cpu": ["CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider"],
    "cuda": ["CUDAExecutionProvider"],
    "default": ort.get_available_providers(),
    "coreml": ["CoreMLExecutionProvider"]
}

AVAILABLE_DEVICE_ALIAS = [
    alias for (alias, pvd_list) in DEVICE_MAPPING.items()
    if pvd_list[0] in ort.get_available_providers()
]

PARTITION_MIN_OVERLAP = 0.2
MULTISCALE_NMS_OVERLAP_THRE = 0.1


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
                 providers_key: str = "default",
                 logger=logger) -> None:
        """Init a YOLOModel that handles YOLO-like inputs and outputs.

        Args:
            weight_path (str): /path/to/the/weight/file.
            dtype (str): model dtype. Should be selected from float32, float16, int8.
            nms (bool, optional): whether to execute non-maximum suppression (NMS) for model outputs.
                If the model is not exported with nms, set to True. Defaults to False.
            warmup (bool, optional): warmup to model before batch processing. Defaults to True.
            pos_thre (float, optional): positive confidence threshold for positive samples. Defaults to 0.25.
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
                be divided into \(2^{1} \times 2^{1} = 2 \times 2\) sub-images and sent to the model. 
                The default value is 2.
            hw_tolerance (float, optional): The max allowed scaling ratio. When running with multiscale mode,
                if diff height-width ratio is larger than excepted, image division will be applied.
            providers_key (str, optional): model provider. Defaults to "default".
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

    def _forward(self, x: np.ndarray) -> tuple:
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

        # for yolo results, [0:4] for pos(xywh), 4 for conf, [5:] for cls_score.
        xywh2xyxy(results[:, :4], inplace=True)
        if self.nms:
            # 只选取统计 conf>thre 及 cls>thre 的结果(done with NMSBoxes)
            res = cv2.dnn.NMSBoxes(bboxes=results[:, :4],
                                   scores=results[:, 4],
                                   score_threshold=self.pos_thre,
                                   nms_threshold=self.nms_thre)
            results = results[list(res)]
        # resize back if necessary
        if self.resize:
            results[:, 0] *= self.scale_w
            results[:, 2] *= self.scale_w
            results[:, 1] *= self.scale_h
            results[:, 3] *= self.scale_h
        # 整数化坐标，类别输出概率矩阵
        result_pos = np.array(results[:, :4], dtype=int)
        # prob以修正分数，得分会很低，因此使用sqrt()的得分修正公式。
        # TODO: 通过优化模型取缔这个tricky的设置。
        result_cls = np.sqrt(
            np.einsum("ab,a->ab", results[:, 5:], results[:, 4]))
        return result_pos, result_cls

    def forward(self, x: np.ndarray):
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

        # 预处理：转换为dtype并归一化[0,1]范围。
        # TODO: 未兼容INT8场合。
        x = x.astype(self.dtype) / 255

        if self.multiscale_pred == 0:
            return self.forward(x)

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
        result_pos = []
        result_cls = []
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
                        clip_pos, clip_cls = self._forward(clip_img)
                        clip_pos[:, 1] += i * h_stride
                        clip_pos[:, 3] += i * h_stride
                        clip_pos[:, 0] += j * w_stride
                        clip_pos[:, 2] += j * w_stride
                        result_pos.append(clip_pos)
                        result_cls.append(clip_cls)
        except Exception as e:
            # 异常跳过
            logger.error(
                f"Exception {e.__repr__()} encountered with calling {self.__class__.__name__}. "
                f"Skip this frame to continue detection...")
            return [], []
        result_pos = np.concatenate(result_pos, axis=0)
        result_cls = np.concatenate(result_cls, axis=0)

        # 重整后 NMS
        res = cv2.dnn.NMSBoxes(
            bboxes=result_pos[:, :4],  # type: ignore
            scores=np.max(result_cls, axis=-1),
            score_threshold=self.pos_thre,
            nms_threshold=MULTISCALE_NMS_OVERLAP_THRE)
        result_pos = result_pos[list(res)]
        result_cls = result_cls[list(res)]

        # 输出前将结果转置回来
        if transpose_flag:
            result_pos = result_pos[:, [1, 0, 3, 2]]

        return result_pos, result_cls


class Backend(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, weight_path, dtype, warmup) -> None:
        pass

    @property
    def input_shape(self):
        pass

    @property
    def input_name(self):
        pass

    @property
    def device(self):
        pass

    @abstractmethod
    def forward(self, x) -> list:
        pass


class ONNXBackend(Backend):

    def __init__(self,
                 weight_path: str,
                 dtype: np.dtype,
                 warmup: bool,
                 providers_key: Optional[str] = None,
                 logger=None) -> None:
        f"""Init a ONNXBackend that use onnxruntime as backend, supporting onnx format weight.
        Args:
            weight_path (str): /path/to/the/weight/file.
            dtype (np.dtype): converted numpy data type.
            warmup (bool, optional): warmup to model before batch processing. Defaults to True.
            providers_key (str, optional): model provider. Defaults to "default".
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
            providers = DEVICE_MAPPING["default"]
        else:
            providers = DEVICE_MAPPING.get(providers_key,
                                           DEVICE_MAPPING["default"])
        self.model_session = ort.InferenceSession(self.weight_path,
                                                  providers=providers)
        self.shapes = [x.shape for x in self.model_session.get_inputs()]
        self.names = [x.name for x in self.model_session.get_inputs()]

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
    def input_shape(self) -> list:
        return self.shapes

    @property
    def input_name(self) -> list:
        return self.names

    @property
    def device(self):
        return self.model_session.get_providers()[0]

    def forward(self, x: Union[np.ndarray, list[np.ndarray]]):
        """ run inference session with given input.

        Args:
            x (Union[np.ndarray, list[np.ndarray]]): input tensor. Its type should varies
                according to the model. For model with multi-inputs, x should be a list of
                tensors; otherwise x should be a tensor.

        Returns:
            list: raw output.
        """
        if self.single_input:
            return self.model_session.run(None, {self.input_name[0]: x})

        return self.model_session.run(None, {
            name: tensor
            for name, tensor in zip(self.input_name, x)
        })



available_models = {"YOLOModel": YOLOModel}
SUFFIX2BACKEND = {"onnx": ONNXBackend}


# TODO: 稍微有点不美观...
def init_model(cfg, **kwargs):
    assert "name" in cfg, "Must specify model name in \"model\"."
    if not cfg.get("name") in available_models:
        raise Exception(f"No model named {cfg.get('name')}.")
    Model = available_models[cfg.get('name')]
    del cfg["name"]
    return Model(**cfg, **kwargs)
