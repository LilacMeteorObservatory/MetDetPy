import cv2
import numpy as np
import onnxruntime as ort

from abc import abstractmethod, ABCMeta
from .MetLog import get_default_logger
from .utils import xywh2xyxy, STR2DTYPE

ort.set_default_logger_severity(3)
logger = get_default_logger()


class YOLOModel(object):

    def __init__(self,
                 weight_path,
                 dtype,
                 nms: bool = False,
                 warmup: bool = True,
                 pos_thre: float = 0.25,
                 nms_thre: float = 0.45,
                 logger=logger) -> None:
        self.weight_path = weight_path
        self.dtype = STR2DTYPE.get(dtype, np.float32)
        self.nms = nms
        self.pos_thre = pos_thre
        self.nms_thre = nms_thre
        self.logger = logger
        self.unchecked = True
        self.resize = False

        # init model
        model_suffix = self.weight_path.split(".")[-1].lower()
        assert model_suffix in SUFFIX2BACKEND, f"Model arch not supported: only support {SUFFIX2BACKEND.keys()}, got {model_suffix}."
        self.BackendCls = SUFFIX2BACKEND[model_suffix]
        self.backend: Backend = self.BackendCls(self.weight_path, self.dtype,
                                                warmup)
        self.logger.info(
            f"Sucessfully load {self.weight_path} on device= {self.backend.device} with Warmup={warmup}."
        )

        # for yolo only first argument is working.
        self.b, self.c, self.h, self.w = self.backend.input_shape[0]
        self.scale_w, self.scale_h = 1, 1

    def forward(self, x):
        # 仅在第一次运行时检查
        if self.unchecked:
            h, w, c = x.shape
            assert c == self.c, "num_channel must match."
            if (h != self.h or w != self.w):
                self.logger.warning(
                    f"Model input shape ({self.h}x{self.w}) is"
                    f"not strictly matched with config ({h}x{w}). "
                    f"Extra resize is applied to avoid error (which may increase time cost.)"
                )
                self.resize = True
                self.scale_h, self.scale_w = h / self.h, w / self.w
            self.unchecked = False

        # resize if necessary
        if self.resize:
            x = cv2.resize(x, (self.w, self.h), interpolation=cv2.INTER_CUBIC)

        # cvt h*w*c to b*c*h*w, range from int8->[0,1]
        x = x.astype(self.dtype).transpose(2, 0, 1)
        x = x[None, ...] / 255
        results = self.backend.forward(x)[0][0]

        # for yolo results, [0:4] for pos(xywh), 4 for conf, [5:] for cls_score.
        xywh2xyxy(results[:, :4], inplace=True)
        if self.nms:
            # 只选取统计conf>thre 及 cls>thre 的结果
            #conf_score = results[:, 4]
            #conf_result = results[conf_score > self.pos_thre]
            #cls_idx = np.sum((conf_result[:, -3:] > self.pos_thre), axis=1)
            #results = conf_result[cls_idx > 0]
            res = cv2.dnn.NMSBoxes(bboxes=results[:, :4],
                                   scores=results[:, 4],
                                   score_threshold=self.pos_thre,
                                   nms_threshold=self.nms_thre)
            results = results[list(res)]
        # TODO: resize back if necessary
        if self.resize:
            results[:,0] *= self.scale_w
            results[:,2] *= self.scale_w
            results[:,1] *= self.scale_h
            results[:,3] *= self.scale_h
        # 整数化坐标，argmax取得类别
        result_pos = np.array(results[:, :4], dtype=int)
        result_cls = np.argmax(results[:, 5:], axis=1)

        
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

    def __init__(self, weight_path, dtype, warmup) -> None:
        self.weight_path = weight_path
        self.dtype = dtype
        # load model
        # TODO: 手动可以强制降级指定运行的设备
        available_providers = ort.get_available_providers()
        self.model_session = ort.InferenceSession(self.weight_path, providers=available_providers)
        self.shapes = [x.shape for x in self.model_session.get_inputs()]
        self.names = [x.name for x in self.model_session.get_inputs()]

        self.single_input = True if len(self.shapes) == 1 else False

        # make sure batch=1
        # Warming up
        if warmup:
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

    def forward(self, x):
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
