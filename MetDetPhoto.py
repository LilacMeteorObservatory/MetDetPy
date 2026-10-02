"""
利用目标检测模型，从单张图像或图像序列批量检测流星的工具。

适用数据：
* 单张图像
* 批量图像（文件夹或 @清单文件）
* 延时视频（照片构成的序列）

## 支持的保存格式
1. 带有标注框的图像
2. 图像和标注文件
3. MDRF形式的摘要
"""
import argparse
import json
import os.path as path
import os
from typing import cast
from time import perf_counter

import cv2
import numpy as np
import tqdm
from numpy.typing import NDArray

from MetLib.fileio import (SUPPORT_ALL_IMG_FORMAT, SUPPORT_COMMON_FORMAT,
                           is_ext_within, load_8bit_image, load_mask,
                           load_raw_with_preprocess, save_path_handler)
from MetLib.imgloader import MultiThreadImgLoader
from MetLib.image_manifest import load_image_manifest
from MetLib.metlog import get_default_logger, set_default_logger
from MetLib.metstruct import MDRF, MockVideoObject, SingleImgRecord
from MetLib.metvisu import (BaseVisuAttrs, ColorTuple, DrawRectVisu,
                            OpenCVMetVisu, SquareColorPair, TextColorPair,
                            TextVisu)
from MetLib.model import AVAILABLE_DEVICE_ALIAS, YOLOModel
from MetLib.onnx_devices import (describe_adapter, discover_dml_adapters,
                                 parse_photo_devices)
from MetLib.photo_inference import PhotoInferencePool, PhotoTask, create_photo_models
from MetLib.utils import (VERSION, exclude_predictions_by_name, get_id2name,
                          parse_resize_param, pt_offset, relative2abs_path)
from MetLib.videoloader import ThreadVideoLoader
from MetLib.videowrapper import OpenCVVideoWrapper

SUPPORT_VIDEO_FORMAT = ["avi", "mp4", "mkv", "mpeg"]
EXCLUDE_LIST = ["PLANE/SATELLITE", "BUGS"]
DEFAULT_COLOR = (64, 64, 64)
DEFAULT_VISUAL_WINDOW_SIZE = [960, 540]
CATE2COLOR_MAPPING: dict[str, ColorTuple] = {
    "METEOR": (0, 255, 0),
    "PLANE/SATELLITE": DEFAULT_COLOR,
    "RED_SPRITE": (0, 0, 255),
    "LIGHTNING": (128, 128, 128),
    "JET": (0, 0, 255),
    "RARE_SPRITE": (0, 0, 255),
    "SPACECRAFT": (255, 0, 255)
}
ID2NAME = get_id2name()


def construct_visu_info(boxes: NDArray[np.int_],
                        preds: NDArray[np.float64],
                        watermark_text: str = ""):
    """构建可视化信息返回串。

    Args:
        img (np.ndarray): background image
        boxes (list[np.ndarray]): boxes
        preds (list[np.ndarray]): pred
        watermark_text (str, optional): watermark. Defaults to "".

    Returns:
        dict: visu_info that can be loaded by MetVisu directly.
    """
    active_meteors: list[SquareColorPair] = []
    score_bg: list[SquareColorPair] = []
    score_text: list[TextColorPair] = []
    for b, p in zip(boxes, preds):
        cate_id = int(np.argmax(p))
        color = CATE2COLOR_MAPPING.get(ID2NAME[cate_id], DEFAULT_COLOR)
        x1, y1, x2, y2 = b
        text = f"{ID2NAME[cate_id]}:{np.max(p):2f}"
        active_meteors.append(
            SquareColorPair(([x1, y1], [x2, y2]), color=color))
        score_bg.append(
            SquareColorPair(
                ([x1, y1], pt_offset((x1, y1), (10 * len(text), -15))),
                color=color))
        score_text.append(
            TextColorPair(text, position=pt_offset((x1, y1), (0, -2))))
    visu_info: list[BaseVisuAttrs] = [
        TextVisu("timestamp",
                 text_list=[TextColorPair(watermark_text)],
                 position="left-bottom",
                 color="white",
                 position_flag=True),
        DrawRectVisu("activate_meteors", pair_list=active_meteors),
        DrawRectVisu("score_bg", pair_list=score_bg, thickness=-1),
        TextVisu("score_text", text_list=score_text, color="white")
    ]
    return visu_info


def _consume_predictions(predictions, results, visual_manager, args, logger,
                         total, image_mode):
    """Keep display, filtering and sparse MDRF output on the main thread."""
    with tqdm.tqdm(total=total, ncols=100) as progress:
        for prediction in predictions:
            task, img = prediction.task, prediction.image
            progress.update(task.index + 1 - progress.n)
            boxes, preds = prediction.boxes, prediction.scores
            if args.visu:
                visual_manager.display_a_frame(
                    img,
                    construct_visu_info(boxes,
                                        preds,
                                        watermark_text=task.source))
                if visual_manager.manual_stop:
                    logger.info('Manual interrupt signal detected.')
                    break
            if args.exclude_noise:
                boxes, preds = exclude_predictions_by_name(
                    boxes, preds, EXCLUDE_LIST)
            if len(boxes) > 0:
                fields = (dict(img_filename=task.source,
                               img_size=list(img.shape[1::-1]))
                          if image_mode else dict(num_frame=task.index))
                record = SingleImgRecord(
                    boxes=[list(map(int, box)) for box in boxes],
                    preds=[ID2NAME[int(np.argmax(pred))] for pred in preds],
                    prob=[
                        f"{pred[int(np.argmax(pred))]:.2f}" for pred in preds
                    ],
                    **fields)
                results.append(record)
                logger.meteor(str(record))
            else:
                logger.debug(
                    f"Input {task.source} detection finished with no result.")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        nargs="?",
        help="path to an image, folder, video, or @image-list.txt.")
    parser.add_argument("--mask", help="path to the mask file.")
    parser.add_argument("--model-path",
                        help="/path/to/the/model",
                        default=None)
    parser.add_argument(
        "--device",
        type=parse_photo_devices,
        default=None,
        help=
        "default, cpu, gpu, gpu,cpu, or explicit list such as dml:0,dml:1 / coreml,cpu."
    )
    parser.add_argument("--list-devices",
                        action="store_true",
                        help="list DML adapters without loading a model.")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help=
        "CPU inference threads: defaults to 0 alone, 1 when mixed with other devices."
    )
    parser.add_argument("--exclude-noise", action="store_true")
    parser.add_argument("--model-type",
                        help="type of the model. Support YOLO.",
                        default="YOLOModel")
    parser.add_argument("--debayer",
                        help="apply debayer to the given image/video.",
                        action="store_true")
    parser.add_argument("--debayer-pattern",
                        help="debayer pattern, like RGGB or BGGR.")
    parser.add_argument("--scale",
                        "-M",
                        type=int,
                        default=2,
                        help="multiscale num.")
    parser.add_argument("--partition",
                        "-P",
                        type=int,
                        default=2,
                        help="partition in pyramid.")
    parser.add_argument("--visu",
                        "-V",
                        action="store_true",
                        help="show detect results.")
    parser.add_argument("--visu-resolution",
                        "-R",
                        type=str,
                        help="detect results showing resolution.")
    parser.add_argument("--save-path",
                        "-S",
                        type=str,
                        help="save path for MDRF.")
    parser.add_argument("--debug",
                        "-D",
                        action="store_true",
                        help="debug mode.")

    args = parser.parse_args(argv)
    if args.num_threads is not None and args.num_threads < 0:
        parser.error("--num-threads must be nonnegative.")
    if args.list_devices:
        print("Installed ONNX devices: " + ", ".join(AVAILABLE_DEVICE_ALIAS))
        try:
            adapters = discover_dml_adapters()
            for adapter in adapters:
                print(describe_adapter(adapter))
            if not adapters:
                print("No DML adapters discovered.")
        except OSError as error:
            print(f"DML adapter discovery failed: {error}")
        return
    if args.target is None:
        parser.error("target is required unless --list-devices is used.")
    list_mode: bool = args.target.startswith("@")
    input_path = args.target[1:] if list_mode else args.target
    suffix = input_path.rsplit(".", 1)[-1].lower()
    sequence_mode = list_mode or os.path.isdir(input_path) or (
        os.path.isfile(input_path) and suffix in SUPPORT_VIDEO_FORMAT)
    device_keys: list[str] = args.device or (["gpu"]
                                             if sequence_mode else ["default"])
    if not sequence_mode and (len(device_keys) > 1 or "gpu" in device_keys):
        parser.error(
            "Multi-device selection is supported only for folders, manifests and timelapse videos."
        )
    for key in device_keys:
        alias = key.partition(":")[0]
        if alias not in {"gpu", "default"
                         } and alias not in AVAILABLE_DEVICE_ALIAS:
            parser.error(f"Requested ONNX device is not installed: {key}")

    if args.model_path is None:
        args.model_path = "./weights/yolov5s_v2.onnx"

    img_list = load_image_manifest(input_path) if list_mode else None
    model_path = relative2abs_path(args.model_path) if not path.isabs(
        args.model_path) else args.model_path
    visu_resolution = parse_resize_param(
        args.visu_resolution, DEFAULT_VISUAL_WINDOW_SIZE
    ) if args.visu_resolution else DEFAULT_VISUAL_WINDOW_SIZE

    set_default_logger(debug_mode=args.debug, work_mode="frontend")
    logger = get_default_logger()

    started = perf_counter()
    logger.start()
    valid_flag = False
    results: list[SingleImgRecord] = []
    video = None
    pool = None
    video_started = False
    try:

        def factory(key, threads):
            return YOLOModel(model_path,
                             dtype="float32",
                             nms=True,
                             warmup=True,
                             logger=logger,
                             providers_key=key,
                             multiscale_pred=args.scale,
                             multiscale_partition=args.partition,
                             num_threads=threads)

        models = create_photo_models(device_keys, factory,
                                     AVAILABLE_DEVICE_ALIAS, logger,
                                     args.num_threads)
        model = models[0][1]
        if list_mode or os.path.isdir(input_path):
            # img folder mode
            img_list: list[str] = img_list if list_mode else [
                os.path.join(input_path, x)
                for x in sorted(cast(list[str], os.listdir(input_path)))
                if is_ext_within(x, SUPPORT_ALL_IMG_FORMAT)
            ]
            visual_manager = OpenCVMetVisu(exp_time=1,
                                           resolution=visu_resolution,
                                           flag=args.visu)
            img_loader = MultiThreadImgLoader(img_list, logger=logger)
            # temp fix: mock video object
            video = MockVideoObject(image_folder=input_path)

            def image_tasks():
                for i in range(len(img_list)):
                    img_path, img = img_loader.pop()
                    if img is None:
                        logger.error(
                            f"Failed to load image {img_path or img_list[i]}.")
                        if img_path is None:
                            break
                        continue
                    yield PhotoTask(i, img_path, img)

            def prepare(image):
                if args.mask:
                    return image * load_mask(args.mask, list(
                        image.shape[1::-1]))
                return image

            pool = PhotoInferencePool(models, prepare=prepare, logger=logger)
            try:
                img_loader.start()
                with pool:
                    predictions = pool.map(image_tasks())
                    try:
                        _consume_predictions(predictions,
                                             results,
                                             visual_manager,
                                             args,
                                             logger,
                                             total=len(img_list),
                                             image_mode=True)
                    finally:
                        predictions.close()
            except (Exception, KeyboardInterrupt) as error:
                logger.error(f"detection terminates caused by: {error!r}")
            finally:
                img_loader.stop()

        elif os.path.isfile(input_path):
            suffix = input_path.split(".")[-1].lower()

            if suffix in SUPPORT_ALL_IMG_FORMAT:
                # img mode
                # temp fix: mock video object
                video = MockVideoObject(image_folder=input_path)
                if is_ext_within(input_path, SUPPORT_COMMON_FORMAT):
                    img = load_8bit_image(input_path)
                else:
                    img = load_raw_with_preprocess(input_path, output_bps=8)
                if img is None:
                    raise ValueError(
                        f"Failed to load image file from {input_path}.")
                mask = load_mask(args.mask, list(img.shape[1::-1]))
                img = img * mask
                visual_manager = OpenCVMetVisu(exp_time=1,
                                               resolution=visu_resolution,
                                               flag=args.visu)
                boxes, preds = model.forward(img)
                if args.exclude_noise:
                    boxes, preds = exclude_predictions_by_name(
                        boxes, preds, EXCLUDE_LIST)
                results = [
                    SingleImgRecord(boxes=[list(map(int, x)) for x in boxes],
                                    preds=[
                                        ID2NAME[int(np.argmax(pred))]
                                        for pred in preds
                                    ],
                                    prob=[
                                        f"{pred[int(np.argmax(pred))]:.2f}"
                                        for pred in preds
                                    ],
                                    img_filename=input_path)
                ]
                logger.info(str(results))
                if args.visu:
                    visu_info = construct_visu_info(boxes,
                                                    preds,
                                                    watermark_text=input_path)
                    visual_manager.display_a_frame(img, visu_info)
                    cv2.waitKey(0)
            elif suffix in SUPPORT_VIDEO_FORMAT:
                # video mode
                video = ThreadVideoLoader(OpenCVVideoWrapper,
                                          input_path,
                                          hwaccel=None,
                                          mask_name=args.mask,
                                          exp_option="real-time",
                                          debayer=args.debayer,
                                          debayer_pattern=args.debayer_pattern,
                                          continue_on_err=True)
                tot_frames = video.iterations
                video.start()
                video_started = True
                visual_manager = OpenCVMetVisu(exp_time=1,
                                               resolution=visu_resolution,
                                               flag=args.visu)
                results = []

                def frame_tasks():
                    for i in range(tot_frames):
                        img = video.pop()
                        if img is None:
                            logger.error(f"Failed to load video frame {i}.")
                            continue
                        yield PhotoTask(i, f"frame {i}", img)

                # VideoLoader has already applied the mask and preprocessing.
                pool = PhotoInferencePool(models, logger=logger)
                try:
                    with pool:
                        predictions = pool.map(frame_tasks())
                        try:
                            _consume_predictions(predictions,
                                                 results,
                                                 visual_manager,
                                                 args,
                                                 logger,
                                                 total=tot_frames,
                                                 image_mode=False)
                        finally:
                            predictions.close()
                except (Exception, KeyboardInterrupt) as error:
                    logger.error(f"detection terminates caused by: {error!r}")
                finally:
                    video.release()
                    video_started = False
            else:
                raise NotImplementedError(
                    f"Unsupport file suffix \"{suffix}\". For now this only support {SUPPORT_VIDEO_FORMAT} and {SUPPORT_ALL_IMG_FORMAT}."
                )
        else:
            raise FileNotFoundError(f"File {input_path} does not exist!")
        valid_flag = True

        # 保存结果
        if valid_flag and args.save_path and video is not None:
            fin_result = MDRF(
                version=VERSION,
                basic_info=video.summary(),
                config=None,
                type="image-prediction" if isinstance(
                    video, MockVideoObject) else "timelapse-prediction",
                anno_size=video.summary().resolution,
                results=results)
            save_path = save_path_handler(args.save_path,
                                          input_path,
                                          ext="json")
            logger.info(f"Result saved to: {save_path}")
            with open(save_path, mode="w", encoding="utf-8") as f:
                json.dump(fin_result.to_dict(),
                          f,
                          ensure_ascii=False,
                          indent=4)

    except Exception as e:
        logger.error(e.__repr__())
    finally:
        if video_started:
            video.release()
        if pool is not None:
            pool.close()
            for line in pool.summary():
                logger.info(line)
        logger.info(
            f"[Photo] total elapsed (including initialization): {perf_counter() - started:.6f}s"
        )
        logger.stop()


if __name__ == "__main__":
    main()
