from typing import Optional, Union
import cv2
import numpy as np
import onnxruntime as ort

from abc import abstractmethod, ABCMeta
from .MetLog import get_default_logger
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
                 providers_key: str = "default",
                 logger=logger) -> None:
        f"""Init a YOLOModel that handles YOLO-like inputs and outputs.

        Args:
            weight_path (str): /path/to/the/weight/file.
            dtype (str): model dtype. Should be selected from {STR2DTYPE.keys()}.
            nms (bool, optional): whether to execute non-maximum suppression (NMS) for model outputs.
                If the model is not exported with nms, set to True. Defaults to False.
            warmup (bool, optional): warmup to model before batch processing. Defaults to True.
            pos_thre (float, optional): positive confidence threshold for positive samples. Defaults to 0.25.
            nms_thre (float, optional): NMS threshold when merging predictions. Defaults to 0.45.
            multiscale_pred (int, optional): the number of prediction scales. When this value is greater
                than 1, the results will be predicted using multi-scale images (similar to a feature pyramid).
                This can improve recall and precision, but may also increase the time cost. Defaults to 1.
            multiscale_partition (int, optional): The number of partitions for multi-scale images at each level.
                For example, at the first sub-level, if the multiscale_partition is set to 2, the image will 
                be divided into \(2^{1} \times 2^{1} = 2 \times 2\) sub-images and sent to the model. 
                The default value is 2.
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
        self.scale_w, self.scale_h = 1, 1

    def forward(self, x: np.ndarray) -> tuple:
        """forward function.

        Args:
            x (np.ndarray): input image. should be 3-channel.

        Returns:
            tuple: a pair of list be like (pred_pos, pred_cls).
        """
        h, w, c = x.shape
        assert c == self.c, "num_channel must match."
        # 仅在第一次运行不匹配时抛出Warning
        if (h != self.h or w != self.w):
            self.resize = True
            self.scale_h, self.scale_w = h / self.h, w / self.w
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

        # cvt h*w*c to b*c*h*w, range from uint8->[0,1]
        x = x.astype(self.dtype).transpose(2, 0, 1)
        x = x[None, ...] / 255
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

    def forward_with_raw_size(self, x, clip_num: Optional[int] = None):
        """forward function with no rescaling. Instead, this function partition the input image to blocks and forward each part.
        
        Results will be recombined then.

        Args:
            x (_type_): _description_

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        h, w, c = x.shape
        assert c == self.c, "num_channel must match."
        # TODO: clip_num功能未完全实现
        h_rep, w_rep = (h - 1) // self.h + 1, (w - 1) // self.w + 1
        h_overlap, w_overlap = (h_rep * self.h - h) // (h_rep - 1), (
            w_rep * self.w - w) // (w_rep - 1)
        result_pos, result_cls = [], []

        for i in range(h_rep):
            for j in range(w_rep):
                clip_img = x[i * self.h - i * h_overlap:(i + 1) * self.h -
                             i * h_overlap, j * self.w -
                             j * w_overlap:(j + 1) * self.w - j * w_overlap]
                clip_pos, clip_cls = self.forward(clip_img)
                clip_pos[:, 1] += i * self.h - i * h_overlap
                clip_pos[:, 3] += i * self.h - i * h_overlap
                clip_pos[:, 0] += j * self.w - j * w_overlap
                clip_pos[:, 2] += j * self.w - j * w_overlap
                result_pos.append(clip_pos)
                result_cls.append(clip_cls)
        result_pos = np.concatenate(result_pos, axis=0)
        result_cls = np.concatenate(result_cls, axis=0)
        # 重整后 NMS
        res = cv2.dnn.NMSBoxes(
            bboxes=result_pos[:, :4],  # type: ignore
            scores=np.max(result_cls, axis=-1),
            score_threshold=self.pos_thre,
            nms_threshold=self.nms_thre)
        result_pos = result_pos[list(res)]
        result_cls = result_cls[list(res)]

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
