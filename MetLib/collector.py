import queue
import threading
from typing import Literal, Optional, Union, cast

import numpy as np

from MetLib.feature import calc_brightness_with_roi

from .metlog import BaseMetLog
from .metstruct import (Box, CollectorCfg, MDTarget, RecheckCfg, RuntimeParams,
                        SingleMDRecord)
from .metvisu import (BaseVisuAttrs, DotColorPair, DrawCircleVisu,
                      DrawRectVisu, SquareColorPair, TextColorPair, TextVisu)
from .model import init_model
from .stacker import max_stacker
from .utils import (ID2NAME, NAME2ID, NUM_CLASS, FloatArray, FloatSeq2D,
                    IntArray, IntSeq2D, _ensure_class_names_loaded,
                    box_matching, color_interpolater, frame2ts, pt_drct,
                    pt_len, pt_len_sqr, pt_offset)
from .videoloader import VanillaVideoLoader

color_mapper = color_interpolater([(128, 128, 128), (128, 128, 128),
                                   (0, 255, 0)])

RECHECK_PADDING_SEC = 0.25
MAX_CONSECUTIVE_FAILURES = 5


class Name2Label(object):
    """类别名称映射到Label的类。
    如果使用自定义的模型，并且输出标签与以下标签不同，
    则需要构建映射标签以确保同一名称映射到相同标签下。

    目前的映射表配置如下：
    
    * 0 - METEOR - 流星
    * 1 - PLANE_SATELLITE 卫星/飞机
    * 2 - RED_SPRITE  普通红色精灵。所有常规精灵目前统一归属该标签下。
    * 3 - LIGHTNING  常规闪电事件。
    * 4 - JET  喷流类精灵。包含巨大喷流，蓝色喷流，次生喷流，蓝色启辉器等。
    * 5 - RARE_SPRITE  稀有类型的精灵。主要是红环，红晕类的大面积黯淡精灵。目前也包含鬼火等样本极少的类别。
    * 6 - SPACECRAFT 人造天体引起的大气景观集合。如发射时的火箭云，航天器再入，燃料排空等。目前类别较少，因此集成。
    * 7 - BUGS  飞虫或小型动物飞行产生的轨迹。
    * 8 - DROPPED 应当被丢弃的类别。(自动生成)
    * 9 - OTHERS  目前未归类的，但能确认并非噪声的响应。(自动生成)
    """
    METEOR = 0
    PLANE_SATELLITE = 1
    RED_SPRITE = 2
    LIGHTNING = 3
    JET = 4
    RARE_SPRITE = 5
    SPACECRAFT = 6
    BUGS = 7

    @staticmethod
    def DROPPED():
        from .utils import get_num_class
        return get_num_class() - 3

    @staticmethod
    def OTHERS():
        from .utils import get_num_class
        return get_num_class() - 2

    @staticmethod
    def BRIGHTNESS_EVENT():
        from .utils import get_num_class
        return get_num_class() - 1


def scale_to(pt: list[int], rescale: list[float]):
    return [int(x * y) for x, y in zip(pt, rescale)]


def create_prob_func(range: FloatArray):
    """A problistic function, return a probilitity instead on 0 or 1.
    Range is designed to be wider.
    For example, if the given range is (a,b), the output func will be like:

            {x/a,  x in [0,a)
    f(x) = { 1 ,   x in [a,b]
            {(2b-x)/b, x in (b,2b]
            {0,    x>2b

    Args:
        range (_type_): _description_

    Returns:
        func: _description_
    """
    a, b = range

    if np.isinf(b):
        def get_prob(x: float):
            if x < a: return x / a
            return 1
    else:
        def get_prob(x: float):
            if x < a: return x / a
            if x <= b: return 1
            if x < 2 * b: return (2 * b - x) / b
            return 0

    return get_prob


class PointList(object):

    def __init__(self) -> None:
        self.pts = np.zeros((0, 2), dtype=np.int32)
        self.frame_num = np.zeros((0, ), dtype=np.int16)

    def append(self, new_pt: IntArray, frame: int):
        if not isinstance(new_pt, np.ndarray):
            new_pt = np.array(new_pt, dtype=int)
        if new_pt.shape == (2, ):
            new_pt = new_pt.reshape(-1, 2)
        self.pts = np.concatenate([self.pts, new_pt], axis=0)
        self.frame_num = np.concatenate(
            [self.frame_num, np.array(frame)], axis=0)

    def extend(self, new_pts: IntSeq2D, frame: int):
        self.pts = np.concatenate([self.pts, np.array(new_pts)], axis=0)
        self.frame_num = np.concatenate(
            [self.frame_num, np.ones((len(new_pts), )) * frame], axis=0)

    def __iter__(self):
        self.iteration = -1
        return self

    def get_pts(self):
        return self.pts

    def __next__(self):
        self.iteration += 1
        if self.iteration == len(self.pts):
            raise StopIteration
        else:
            return self.pts[self.iteration]

    def get_pts_as_list(self) -> list[list[int]]:
        return [[int(x[0]), int(x[1])] for x in self.pts]

    def __getitem__(self, i: int) -> IntSeq2D:
        return self.pts[i]

    def __len__(self):
        return len(self.pts)


class MeteorSeries(object):
    """用于整合检测结果，排异和给出置信度的流星序列。

    Args:
        object (_type_): _description_
    """

    def __init__(self, start_frame: int, cur_frame: int, init_pts: IntSeq2D,
                 max_acceptable_dist: int, max_acti_frame: int,
                 cate_prob: FloatArray, fps: float, runtime_size: list[int]):
        """_summary_

        Args:
            start_frame (_type_): _description_
            cur_frame (_type_): _description_
            init_box (_type_): shape [n, 2]
            max_acceptable_dist (_type_): _description_
            max_acti_frame (_type_): _description_
            cate_prob (_type_): _description_
        
        MeteorSeries Property:
            start_frame [int] 起始帧
            last_motion_frame [int] 最后运动帧（目标扩展运动范围的最后时刻）
            last_activate_frame [int] 最后响应帧
        """
        assert len(init_pts) in (
            3, 5
        ), f"invalid init_pts length: should be 3 but {len(init_pts)} got."
        self.coord_list = PointList()
        self.center_list = PointList()
        self.drct_list: list[float] = []
        self.coord_list.extend(init_pts, cur_frame)
        self.center_list.extend(np.mean(init_pts, axis=0)[None, :], cur_frame)
        self.drct_list.append(pt_drct(init_pts[0], init_pts[1]))
        self.start_frame = start_frame
        self.last_motion_frame = cur_frame
        self.last_activate_frame = cur_frame
        self.max_acti_frame = max_acti_frame
        self.max_acceptable_dist = max_acceptable_dist
        self.count = 1
        self.cate_prob = np.array(cate_prob, copy=True)
        self._score_cache_count = -1
        self._score_cache_value = 0.0
        self.fps = fps
        self.runtime_length = max(runtime_size)
        self.range = ([2**16, 2**16], [-2**16, -2**16])
        self.calc_new_range(init_pts)

    @property
    def drst_std(self):
        if len(self.drct_list) == 0: return 0
        drct_copy = np.array(self.drct_list.copy())
        std1 = np.std(np.sort(drct_copy)[:-1]) if len(
            drct_copy) >= 3 else np.std(drct_copy)
        drct_copy[drct_copy > np.pi / 2] -= np.pi
        std2 = np.std(np.sort(drct_copy)[:-1]) if len(
            drct_copy) >= 3 else np.std(drct_copy)
        return cast(float, min(std1, std2))

    @property
    def drst_cv(self) -> float:
        """Circular variance of direction angles. Range [0, 1], 0 = perfectly straight."""
        if len(self.drct_list) == 0:
            return 0.0
        angles = np.array(self.drct_list) * 2
        return float(1 - abs(np.mean(np.exp(1j * angles))))

    @property
    def cate(self) -> int:
        return np.argmax(self.cate_prob, axis=0)

    @property
    def duration(self) -> int:
        """片段的持续帧跨度（last_activate_frame - start_frame）。

        start_frame 为合并窗口起始估计，非首次观测帧，因此 duration 表示时间跨度而非观测计数。
        计算速度时应使用 fix_motion_duration。
        """
        return self.last_activate_frame - self.start_frame

    @property
    def fix_duration(self) -> float:
        """流星序列的真实持续时间（单位为秒）。

        Returns:
            float: _description_
        """
        return self.duration / self.fps

    @property
    def fix_motion_duration(self) -> float:
        """流星序列的真实运动时间（单位为秒）。
        """
        return (self.last_motion_frame - self.start_frame) / self.fps

    @property
    def sort_range(self):
        """range的增强版，按照时间顺序给出起止点组合
        """
        [x0, y0], [x1, y1] = self.range
        e_x, e_y = self.coord_list[int(np.argmin(self.coord_list.frame_num))]
        l_x, l_y = self.coord_list[int(np.argmax(self.coord_list.frame_num))]
        if e_x > l_x:
            x0, x1 = x1, x0
        if e_y > l_y:
            y0, y1 = y1, y0
        return [x0, y0], [x1, y1]

    @property
    def dist(self) -> float:
        pt1, pt2 = self.range
        return pt_len(pt1, pt2)

    @property
    def fix_dist(self):
        """返回流星序列的真实长度。单位为移动距离（长边画幅移动比例），数值会 x100 以放缩到常规数值范围。

        Returns:
            _type_: _description_
        """
        return self.dist / self.runtime_length * 100

    @property
    def speed(self) -> float:
        """返回流星序列的平均速度。其中距离通过直接求最大跨度获得，时间仅使用运动期间的时长。
        
        NOTE: `speed` 属性是相对的（运行时分辨率，时间长为帧）。真实速度需要使用 `fix_speed` 接口。

        Returns:
            _type_: _description_
        """
        return self.dist / (self.last_motion_frame - self.start_frame + 1e-6)

    @property
    def fix_speed(self) -> float:
        """返回流星序列的真实平均速度。
        
        运行速度单位为移动距离（长边画幅移动比例）/时间（秒），数值会 x100 以放缩到常规数值范围。

        Returns:
            float: _description_
        """
        return self.speed * self.fps / self.runtime_length * 100

    def get_met_attr(self, decimals: int = 3) -> MDTarget:
        """
        将自身转换为 MDTarget 结构体。
        score 和 real_dist 不由本身不填充。
        TODO: 是不是该在内部完成，否则逻辑非常冗杂。
        
        NOTE: 部分数值会被截断以适应输出格式。

        Returns:
            MDTarget: 转换后结构体。
        """
        pt1, pt2 = self.sort_range
        dist: float = pt_len(pt1, pt2)

        return MDTarget(start_time=frame2ts(self.start_frame, self.fps),
                        start_frame=self.start_frame,
                        end_time=frame2ts(self.last_motion_frame, self.fps),
                        last_activate_frame=self.last_activate_frame,
                        last_activate_time=frame2ts(self.last_activate_frame,
                                                    self.fps),
                        duration=self.duration,
                        speed=np.round(self.speed, decimals),
                        dist=np.round(dist, decimals),
                        fix_dist=np.round(self.fix_dist, decimals),
                        fix_speed=np.round(self.fix_speed, decimals),
                        fix_motion_duration=np.round(self.fix_motion_duration,
                                                     decimals),
                        fix_duration=np.round(self.fix_duration, decimals),
                        num_pts=len(self.coord_list),
                        category=ID2NAME[self.cate],
                        pt1=pt1,
                        pt2=pt2,
                        center_point_list=self.center_list.get_pts_as_list(),
                        drct_loss=np.round(self.drst_std, 3),
                        drct_cv=np.round(self.drst_cv, 4),
                        score=-1,
                        real_dist=-1)

    def calc_new_range(self, pts: IntSeq2D) -> None:
        """基于输入的新点集，更新该 MeteorSeries 的范围值 (self.range). 

        Args:
            pts (list): 点集合
        """
        self.range = [
            min(int(min([pt[0] for pt in pts])), self.range[0][0]),
            min(int(min([pt[1] for pt in pts])), self.range[0][1])
        ], [
            max(int(max([pt[0] for pt in pts])), self.range[1][0]),
            max(int(max([pt[1] for pt in pts])), self.range[1][1])
        ]

    def update(self, new_frame: int, new_box: IntSeq2D, new_cate: FloatArray):
        """为序列更新新的响应

        Args:
            new_frame (_type_): _description_
            new_box (_type_): _description_
            new_cate (_type_): _description_
        """
        (x1, y1), (x2, y2) = self.range
        assert len(new_box) in (
            3,
            5), f"invalid init_pts length: should be 3 but {len(new_box)} got."
        # 超出区域时，更新last_motion_frame; 否则仅更新last_activate_frame
        for pt in new_box:
            if not ((x1 <= pt[0] <= x2) and (y1 <= pt[1] <= y2)):
                self.last_motion_frame = new_frame
                break
        self.last_activate_frame = new_frame
        self.coord_list.extend(new_box, new_frame)
        self.center_list.extend(np.mean(new_box, axis=0)[None, :], new_frame)
        # range由calc_new_range更新，除去init外每次仅在update时更新
        self.calc_new_range(new_box)
        self.drct_list.append(pt_drct(new_box[0], new_box[1]))
        self.cate_prob += new_cate
        self.count += 1

    def may_in_series(self, pts: IntSeq2D, cur_frame: int):
        # 策略一：最后近邻法（对于有尾迹的判断不准确）
        #if pt_len(self.box2coord(new_box)+self.coord_list[-1])<self.max_acceptable_dist:
        #    return True
        # 策略二：近邻法（对于距离中间点近的，采取收入但不作为边界点策略）
        first = np.where(self.coord_list.frame_num >= cur_frame -
                         self.max_acti_frame)[0]
        first = len(self.coord_list.frame_num) if len(first) == 0 else first[0]
        for tgt_pt in pts:
            for in_pt in self.coord_list[first:]:
                if pt_len_sqr(tgt_pt, in_pt) < self.max_acceptable_dist:
                    return True
        return False


class MeteorCollector(object):
    """
    全局的流星统计模块。用于记录和管理所有的响应，整合成正在发生（或已经结束）的检测序列，执行必要的重校验。
    """

    def __init__(self, collector_cfg: CollectorCfg,
                 runtime_param: RuntimeParams,
                 video_loader: Optional[VanillaVideoLoader],
                 logger: BaseMetLog) -> None:
        self.min_len = collector_cfg.meteor_cfg.min_len
        self.max_interval = collector_cfg.meteor_cfg.max_interval * runtime_param.fps
        self.max_acti_frame = int(collector_cfg.meteor_cfg.max_interval *
                                  runtime_param.fps)
        self.det_thre = collector_cfg.meteor_cfg.det_thre

        # merge_dist_sqr: 序列归并的最大距离平方阈值，乘以 exp_frame 做帧合并后距离放大补偿
        meteor_cfg = collector_cfg.meteor_cfg
        if meteor_cfg.merge_dist_sqr is not None:
            base_dist_sqr = meteor_cfg.merge_dist_sqr
        elif meteor_cfg.thre2 is not None:
            # deprecated compat path; will be removed in v3.0.0
            import warnings
            warnings.warn(
                "Config field 'thre2' is deprecated, use 'merge_dist_sqr' instead.",
                DeprecationWarning, stacklevel=2)
            base_dist_sqr = meteor_cfg.thre2
        else:
            raise ValueError("Either 'merge_dist_sqr' or 'thre2' must be specified in meteor_cfg.")
        self.merge_dist_sqr = base_dist_sqr * runtime_param.exp_frame
        self.runtime_size = runtime_param.runtime_size
        self.active_meteor: list[MeteorSeries] = []
        self.cur_frame = 0
        self.eframe = runtime_param.exp_frame
        self.fps = runtime_param.fps
        self.time_prob_func = create_prob_func(
            collector_cfg.meteor_cfg.time_range)
        self.speed_prob_func = create_prob_func(
            collector_cfg.meteor_cfg.speed_range)
        self.len_prob_func = create_prob_func((self.min_len, np.inf))
        self.drct_prob_func = create_prob_func(
            collector_cfg.meteor_cfg.drct_range)
        self.logger = logger

        recheck_thre = collector_cfg.meteor_cfg.recheck_threshold
        self.recheck_threshold = (recheck_thre if recheck_thre is not None
                                  else self.det_thre * 0.5)

        clip_merge_sec = collector_cfg.meteor_cfg.clip_merge_interval
        clip_merge_interval = (clip_merge_sec * runtime_param.fps
                               if clip_merge_sec is not None
                               else self.max_interval)

        # Init Exporter
        self.met_exporter = MetExporter(collector_cfg.recheck_cfg,
                                        runtime_param,
                                        video_loader=video_loader,
                                        logger=logger,
                                        clip_merge_interval=clip_merge_interval,
                                        det_thre=self.det_thre)

    def update(self, cur_frame: int, lines: IntSeq2D, cates: FloatSeq2D):
        """
        更新流星序列的主要函数。

        原则上可以在有新响应时更新，实际为了报告效率，可以无流星时每5-10s执行一次。

        Args:
            cur_frame (_type_): _description_
            lines (_type_): _description_
        """
        self.cur_frame = cur_frame

        # 1. 收集超时序列
        keep_list: list[MeteorSeries] = []
        drop_list: list[MeteorSeries] = []
        for ms in self.active_meteor:
            if self.cur_frame - ms.last_activate_frame >= self.max_interval:
                if self._should_keep(ms):
                    keep_list.append(ms)
                else:
                    drop_list.append(ms)

        # 2. 过滤 active_meteor
        remove_set = set(id(ms) for ms in drop_list + keep_list)
        self.active_meteor = [ms for ms in self.active_meteor if id(ms) not in remove_set]

        # 3. 计算 nearest_active_start（基于过滤后的 active）
        nearest_active_start = self._calc_nearest_active_start()

        # 4. 发送超时序列
        exported = False
        for ms in keep_list:
            attr = self.get_met_attr(ms)
            if attr is None:
                continue
            self.met_exporter.export(
                self.met_exporter.ACTIVE_FLAG,
                [attr], cur_frame, nearest_active_start)
            exported = True
        for ms in drop_list:
            attr = self.get_met_attr(ms)
            if attr is None:
                continue
            self.met_exporter.export(
                self.met_exporter.DROP_FLAG,
                [attr], cur_frame, nearest_active_start)
            exported = True

        # 5. 心跳：确保 Exporter 感知时间推进
        if not exported:
            self.met_exporter.export(
                self.met_exporter.DROP_FLAG, [], cur_frame, nearest_active_start)

        if len(cates) == 0:
            return
        # 做合并
        num_activate = len(self.active_meteor)
        cate_ids = cast(list[int], np.argmax(np.array(cates), axis=1))
        for line_pts, cate_id, cate_prob in zip(lines, cate_ids, cates):
            # 如果某一序列已经开始，则可能是其中间的一部分。
            # 考虑到基本不存在多个流星交接的情况，如果属于某一个，则直接归入即可。
            # TODO: cur_frame+-eframe fixed!!
            # 对于直线类型（流星，飞机），使用头尾及中间点作为点集
            # 对于面积类型（未知类别，闪电，精灵），使用边界点及中心点作为点集
            # TODO: 目前使用硬编码。未来优化。
            if cate_id in [Name2Label.METEOR, Name2Label.PLANE_SATELLITE]:
                line = cast(IntSeq2D,
                            np.array([
                                line_pts[:2], line_pts[2:],
                                (line_pts[:2] + line_pts[2:]) // 2
                            ]))  # type: ignore
            else:
                x1, y1, x2, y2 = line_pts
                # 有点奇怪
                # 此处保留顺序是因为计算方差需要，但对直线类，面积的应该不能参与方差计算。TODO: 这个要考量下。
                line = cast(
                    IntSeq2D,
                    np.array([[x1, y1], [x2, y2], [x2, y1], [x1, y2],
                              [int((x1 + x2) / 2),
                               int((y1 + y2) / 2)]]))
            is_in_series = False
            for ms in self.active_meteor[:num_activate]:
                is_in = ms.may_in_series(line, cur_frame)
                if is_in:
                    ms.update(self.cur_frame, line, new_cate=cate_prob)
                    is_in_series = True
                    break
            # 如果不属于已存在的序列，则为其构建新的序列开头
            if is_in_series:
                continue
            self.active_meteor.append(
                MeteorSeries(max(self.cur_frame - self.eframe, 0),
                             self.cur_frame,
                             line,
                             max_acceptable_dist=self.merge_dist_sqr,
                             max_acti_frame=self.max_acti_frame,
                             cate_prob=cate_prob,
                             fps=self.fps,
                             runtime_size=self.runtime_size))

    def visu(self, frame_num: int):
        active_meteors: list[SquareColorPair] = []
        active_pts: list[DotColorPair] = []
        score_text: list[TextColorPair] = []
        score_bg: list[SquareColorPair] = []
        for ms in self.active_meteor:
            pt1, pt2 = ms.range
            color = color_mapper(self.prob_meteor(ms))

            active_meteors.append(
                SquareColorPair(dot_pair=(pt1, pt2), color=color))

            # 只打印最近的响应点
            first = np.where(ms.coord_list.frame_num >= frame_num -
                             self.max_acti_frame)[0]
            first = len(
                ms.coord_list.frame_num) if len(first) == 0 else first[0]
            for pts in ms.coord_list[first:]:
                pt_x, pt_y = pts
                active_pts.append(DotColorPair(dot=(pt_x, pt_y), color=color))

            # print score
            pt1 = [min(pt1[0], pt2[0]), min(pt1[1], pt2[1])]
            if pt1[1] <= 15: pt1[1] = max(pt1[1], pt2[1]) + 15
            word_length = len(f"{ID2NAME[ms.cate]}:{self.prob_meteor(ms):.2f}")
            score_bg.append(
                SquareColorPair(dot_pair=(pt1,
                                          pt_offset(pt1,
                                                    (10 * word_length, -15))),
                                color=color))
            score_text.append(
                TextColorPair(
                    text=f"{ID2NAME[ms.cate]}:{self.prob_meteor(ms):.2f}",
                    position=pt_offset(pt1, (0, -2))))

        ret: list[BaseVisuAttrs] = [
            DrawRectVisu("active_meteors", pair_list=active_meteors),
            DrawCircleVisu("active_pts",
                           dot_list=active_pts,
                           radius=2,
                           thickness=-1),
            TextVisu("score_text", text_list=score_text, color="white"),
            DrawRectVisu("score_bg", pair_list=score_bg, thickness=-1)
        ]

        return ret

    def _should_keep(self, ms: MeteorSeries) -> bool:
        """判断过期序列是否应保留（送入 Exporter 做 recheck）而非直接丢弃。

        三层过滤：
        1. 单帧响应直接丢弃（高频噪声/卫星闪烁，占比极大）
        2. 无 recheck 时，不确定类别（OTHERS/PLANE）没有后置验证能力，直接丢弃
        3. 前置分类器存在误判，recheck_threshold 低于 det_thre 以放宽送检门槛；
           该阈值为折中：过低则 recheck 计算量过大，过高则漏检前置误判的正样本。
        """
        if ms.count <= 1:
            return False

        if not self.met_exporter.recheck:
            if ms.cate in [Name2Label.OTHERS(), Name2Label.PLANE_SATELLITE]:
                return False

        return self.prob_meteor(ms) > self.recheck_threshold

    def _calc_nearest_active_start(self) -> Optional[int]:
        """返回当前 active_meteor 中"有潜力"序列的最早 start_frame，用于通知 Exporter 是否有潜在的同 clip 候选。"""
        candidates = [ms.start_frame for ms in self.active_meteor
                      if self.prob_meteor(ms) > self.det_thre / 2]
        return min(candidates) if candidates else None

    def clear(self):
        """将所有残留的活跃序列做最终判定并导出，然后结束导出线程。
        应当在结束时仅调用一次。
        """
        for ms in self.active_meteor:
            attr = self.get_met_attr(ms)
            if attr is None:
                continue
            if self._should_keep(ms):
                self.met_exporter.export(
                    self.met_exporter.ACTIVE_FLAG,
                    [attr], self.cur_frame, None)
            else:
                self.met_exporter.export(
                    self.met_exporter.DROP_FLAG,
                    [attr], self.cur_frame, None)
        self.active_meteor.clear()

        self.met_exporter.export(self.met_exporter.END_FLAG, [], self.cur_frame, None)
        self.met_exporter.export_loop.join()

    def prob_meteor(self, met: MeteorSeries) -> float:
        # 用于估计met实例属于流星序列的概率。
        # 缓存策略：缓存跟随 MeteorSeries 生命周期，count 不变则结果不变。
        if met._score_cache_count == met.count:
            return met._score_cache_value
        score = self._compute_score(met)
        met._score_cache_count = met.count
        met._score_cache_value = score
        return score

    def _compute_score(self, met: MeteorSeries) -> float:
        # 计分规则：当属于流星时，按照流星规则统计；当不属于流星时，按照所属类别的最大概率统计。
        # TODO: 可能是不完善的。需要观察验证。
        if met.cate == Name2Label.METEOR:
            # 对短样本实现一定的宽容
            len_prob = self.len_prob_func(met.dist)
            # 排除总时长过长/过短
            time_prob = self.time_prob_func(met.fix_duration)
            # 排除速度过快/过慢
            speed_prob = self.speed_prob_func(met.fix_speed)
            # 计算直线情况
            drct_prob = self.drct_prob_func(met.drst_std)
            return time_prob * speed_prob * len_prob * drct_prob
        else:
            if np.any(np.isnan(met.cate_prob)):
                self.logger.warning(
                    f"NaN detected in cate_prob: {met.cate_prob}. "
                    f"Dropping this meteor series.")
                return 0.0
            return met.cate_prob[met.cate] / met.count

    def get_met_attr(self, met: MeteorSeries) -> Optional[MDTarget]:
        """将met的点集序列转换为属性字典。

        Returns:
            Optional[MDTarget]: 转换后结构体；若 score 异常则返回 None（on-error-drop）。
        """
        score = self.prob_meteor(met)
        if not np.isfinite(score):
            self.logger.warning(
                f"Non-finite score for series at frame {met.start_frame}, dropping.")
            return None
        met_target = met.get_met_attr()
        met_target.score = np.round(score, 2)
        return met_target

    def frame2ts(self, frame: int) -> str:
        return frame2ts(frame, self.fps)


class MetExporter(object):
    """用于管理输出的具体格式，重校验。

    Args:
        object (_type_): _description_

    Raises:
        KeyError: _description_

    Returns:
        _type_: _description_
    """
    END_FLAG = "END_FLAG"
    DROP_FLAG = "DROP_FLAG"
    ACTIVE_FLAG = "ACTIVE_FLAG"
    FLAG_TYPE_ALIAS = Union[Literal["END_FLAG"], Literal["DROP_FLAG"],
                            Literal["ACTIVE_FLAG"]]
    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self, recheck_cfg: RecheckCfg, runtime_param: RuntimeParams,
                 video_loader: Optional[VanillaVideoLoader],
                 logger: BaseMetLog, clip_merge_interval: float,
                 det_thre: float) -> None:
        self.queue: queue.Queue[tuple[str, list[MDTarget], int, Optional[int]]] = queue.Queue()
        self.recheck = recheck_cfg.switch
        self.positive_cates: list[str] = runtime_param.positive_category_list
        self.positive_cate_ids: list[int] = [
            NAME2ID[cate] for cate in self.positive_cates if cate in NAME2ID
        ]
        self.logger = logger
        self.clip_merge_interval = clip_merge_interval
        self.det_thre = det_thre
        self.fps = runtime_param.fps
        if self.recheck:
            self.recheck_loader = video_loader
            self.recheck_model = init_model(recheck_cfg.model,
                                            logger=self.logger)
            self.recheck_padding = int(RECHECK_PADDING_SEC * self.fps)
        # Rescale: 用于将结果放缩回原始分辨率的放缩倍率。
        self.raw_size = runtime_param.raw_size
        self.rescale_ratio = [
            x / y
            for x, y in zip(runtime_param.raw_size, runtime_param.runtime_size)
        ]
        self.export_loop = threading.Thread(target=self.loop, daemon=True)
        self.export_loop.start()
        self.meteor_list: list[SingleMDRecord] = []
        self.pending_confirmed: list[MDTarget] = []
        self.last_seen_frame: int = 0
        self.nearest_active_start: Optional[int] = None

    def export(self, flag: FLAG_TYPE_ALIAS, data: list[MDTarget],
               cur_frame: int = 0, nearest_active_start: Optional[int] = None):
        self.queue.put((flag, data, cur_frame, nearest_active_start))

    def loop(self):
        consecutive_failures = 0
        flag, data, cur_frame, nearest = self.queue.get()
        while flag in [self.ACTIVE_FLAG, self.DROP_FLAG]:
            self.last_seen_frame = cur_frame
            self.nearest_active_start = nearest
            try:
                self._try_flush_pending()
                self._process_message(flag, data)
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                self.logger.error(
                    f"Exporter batch failed ({consecutive_failures}/"
                    f"{self.MAX_CONSECUTIVE_FAILURES}): {e}")
                if consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    self.logger.error(
                        "Exporter loop aborted due to repeated failures.")
                    break
            flag, data, cur_frame, nearest = self.queue.get()
        # END_FLAG: flush all remaining pending
        if flag == self.END_FLAG:
            self._flush_pending()
        elif flag not in [self.ACTIVE_FLAG, self.DROP_FLAG]:
            raise KeyError(
                f"Unexpected flag received. Expect [{self.ACTIVE_FLAG},"
                f"{self.DROP_FLAG},{self.END_FLAG}], got {flag} instead.")

    def _try_flush_pending(self):
        """检查 pending 是否满足 flush 条件并执行。"""
        if not self.pending_confirmed:
            return
        last_pending_frame = self.pending_confirmed[-1].last_activate_frame
        time_exceeded = (self.last_seen_frame - last_pending_frame
                         > self.clip_merge_interval)
        no_active_candidate = (
            self.nearest_active_start is None
            or self.nearest_active_start - last_pending_frame
            > self.clip_merge_interval)
        if time_exceeded and no_active_candidate:
            self._flush_pending()

    def _flush_pending(self):
        """将 pending_confirmed 合并输出并清空。"""
        if not self.pending_confirmed:
            return
        self.pending_confirmed.sort(key=lambda t: t.start_frame)
        final_list = self.merge_targets_by_time(self.pending_confirmed)
        for met in final_list:
            met = self.rescale(met)
            self.meteor_list.append(met)
            self.logger.meteor(met.to_json(full=False))
        self.pending_confirmed.clear()

    def _process_message(self, flag: str, data: list[MDTarget]):
        if flag == self.DROP_FLAG:
            for ms_attr in data:
                # Drop类标签修正
                ms_attr.category = ID2NAME[Name2Label.DROPPED()]
                output_dict = SingleMDRecord.from_target(
                    ms_attr, self.raw_size)
                output_dict = self.rescale(output_dict)
                self.meteor_list.append(output_dict)
                self.logger.dropped(output_dict.to_json(full=False))
        else:
            # ACTIVE_FLAG: 逐 target 独立复检
            for ms_attr in data:
                if self.recheck:
                    result = self.recheck_single_target(ms_attr)
                    if result is not None:
                        self.pending_confirmed.append(result)
                    else:
                        # 置信度不足的正样本类别，在输出前重置为 OTHERS
                        if ms_attr.category in self.positive_cates:
                            ms_attr.category = ID2NAME[Name2Label.OTHERS()]
                        output_dict = SingleMDRecord.from_target(
                            ms_attr, self.raw_size)
                        output_dict = self.rescale(output_dict)
                        self.meteor_list.append(output_dict)
                        self.logger.dropped(output_dict.to_json(full=False))
                else:
                    self.pending_confirmed.append(ms_attr)

    def rescale(self, meteor_dict: SingleMDRecord) -> SingleMDRecord:
        """将复合的meteor_dict中的所有target的起止坐标和距离映射回真实分辨率下。

        Args:
            meteor_dict (dict): 复合的meteor_dict，其target参数为一个列表，包含若干个流星片段。

        Returns:
            dict: 处理后的meteor_dict。
        """
        for single_meteor in meteor_dict.target:
            single_meteor.pt1 = scale_to(single_meteor.pt1, self.rescale_ratio)
            single_meteor.pt2 = scale_to(single_meteor.pt2, self.rescale_ratio)
            single_meteor.real_dist = single_meteor.dist * max(
                self.rescale_ratio)
            for i in range(len(single_meteor.center_point_list)):
                single_meteor.center_point_list[i] = scale_to(
                    single_meteor.center_point_list[i], self.rescale_ratio)
        return meteor_dict

    def recheck_single_target(self, target: MDTarget) -> Optional[MDTarget]:
        """对单个 target 执行独立复检。

        Returns:
            通过复检的 target（可能更新了 category/score），或 None 表示未通过。
        """
        # BRIGHTNESS_EVENT 类别豁免 recheck：该类别由 BrightnessDetector 产生，
        # 属于 DL recheck 模型的域外分布，强制 recheck 会导致误丢弃。
        brightness_event_name = ID2NAME[Name2Label.BRIGHTNESS_EVENT()]
        if target.category == brightness_event_name:
            return target

        assert self.recheck_loader is not None
        stacked_img = max_stacker(
            video_loader=self.recheck_loader,
            start_frame=max(0, target.start_frame - self.recheck_padding),
            end_frame=min(target.last_activate_frame + self.recheck_padding,
                          self.recheck_loader.video_total_frames - 1),
            logger=self.logger)

        if stacked_img is None:
            self.logger.error(
                "Failed to get stacked img. Target will pass without recheck."
                f" start_frame={target.start_frame};"
                f" last_activate_frame={target.last_activate_frame}")
            return target

        bbox_list, score_list = self.recheck_model.forward(stacked_img)
        target_bbox = [[*target.pt1, *target.pt2]]
        matched_pairs = box_matching(bbox_list, target_bbox)  # type: ignore

        if len(matched_pairs) == 0:
            return None

        l, _ = matched_pairs[0]
        label = np.argmax(score_list[l, :], axis=0)
        score = score_list[l, label]

        target.category = ID2NAME.get(label, ID2NAME[Name2Label.OTHERS()])
        target.raw_score = target.score
        target.recheck_score = round(score.astype(np.float64), ndigits=3)

        # 当预测为流星时，求分数均值作为最终得分；否则直接使用模型得分。
        # TODO: 该逻辑仅在前置分类器为规则分类器时生效。未来预计引入前置的机器学习分类器输出多类别分数。
        if label == Name2Label.METEOR:
            mge_score = (target.recheck_score + target.raw_score) / 2
        else:
            mge_score = score.astype(np.float64)
        target.score = np.round(mge_score, 2)

        # label为置信流星，或者为positive_cate_ids中其他类别时，才其加入到正输出中。
        if (label != Name2Label.METEOR and label
                in self.positive_cate_ids) or (label == Name2Label.METEOR and
                                               target.score >= self.det_thre):
            sure_box = Box.from_pts(target.pt1, target.pt2)
            r_brightness = calc_brightness_with_roi(stacked_img, sure_box)
            target.relative_brightness = round(r_brightness, ndigits=3)
            target.aesthetic_score = round(target.score * target.fix_dist *
                                           target.relative_brightness,
                                           ndigits=3)
            return target
        else:
            # 流星类被丢弃时需要重新标记为 DROPPED
            if label == Name2Label.METEOR:
                target.category = ID2NAME[Name2Label.DROPPED()]
            return None

    def merge_targets_by_time(self,
                              targets: list[MDTarget]) -> list[SingleMDRecord]:
        """将通过复检的 targets 按时间邻近合并为输出格式。"""
        if not targets:
            return []
        result: list[SingleMDRecord] = []
        current = SingleMDRecord.from_target(targets[0], self.raw_size)
        for t in targets[1:]:
            if (current.end_frame is not None
                    and t.start_frame < current.end_frame + self.clip_merge_interval):
                if t.last_activate_frame > (current.end_frame or 0):
                    current.end_frame = t.last_activate_frame
                    current.end_time = t.last_activate_time
                current.target.append(t)
            else:
                result.append(current)
                current = SingleMDRecord.from_target(t, self.raw_size)
        result.append(current)
        return result
