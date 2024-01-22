import threading
import queue
import json

import numpy as np

from .Model import init_model
from .Stacker import max_stacker
from .utils import (color_interpolater, drct, drct_line, frame2ts, lineset_nms,
                    save_img, pt_len_xy, pt_offset, box_matching, ID2NAME)

color_mapper = color_interpolater([[128, 128, 128], [128, 128, 128],
                                   [0, 255, 0]])


class Name2Label(object):
    UNKNOWN_AREA = -1
    METEOR = 0
    PLANE = 1
    RED_SPRITE = 2
    LIGHTNING = 3


def scale_to(pt, rescale):
    return [int(x * y) for x, y in zip(pt, rescale)]


def create_prob_func(range):
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

    def get_prob(x):
        if x < a: return x / a
        if a <= x <= b: return 1
        if x < 2 * b: return (2 * b - x) / b
        return 0

    return get_prob


class MeteorCollector(object):
    """
    全局的流星统计模块。用于记录和管理所有的响应，整合成正在发生（或已经结束）的检测序列，执行必要的重校验。
    """

    def __init__(self, meteor_cfg, eframe, fps, runtime_size, raw_size,
                 recheck_cfg, video_loader, logger) -> None:
        self.min_len = meteor_cfg.min_len
        self.max_interval = meteor_cfg.max_interval * fps
        self.max_acti_frame = meteor_cfg.max_interval * fps
        self.det_thre = meteor_cfg.det_thre
        self.thre2 = meteor_cfg.thre2 * eframe
        self.active_meteor = [
            MeteorSeries(np.inf, np.inf, [-100, -100, -101, -101], np.nan,
                         np.nan, "None")
        ]
        self.waiting_meteor = []
        self.ended_meteor = []
        self.cur_frame = 0
        self.eframe = eframe
        self.fps = fps
        self.raw_size = raw_size
        # 调整time的验证下界
        # TODO: 梳理一下time_range相关逻辑
        meteor_cfg.time_range[0] *= fps
        meteor_cfg.time_range[1] *= fps
        meteor_cfg.time_range[0] = max(meteor_cfg.time_range[0],
                                       int(4 * self.eframe + 2))
        self.time_prob_func = create_prob_func(meteor_cfg.time_range)
        self.speed_prob_func = create_prob_func(meteor_cfg.speed_range)
        self.len_prob_func = create_prob_func((self.min_len, np.inf))
        self.drct_prob_func = create_prob_func(meteor_cfg.drct_range)

        # Init Exporter
        self.met_exporter = MetExporter(runtime_size=runtime_size,
                                        raw_size=raw_size,
                                        recheck_cfg=recheck_cfg,
                                        video_loader=video_loader,
                                        logger=logger,
                                        max_interval=self.max_interval)

        # 定义可视化接口字段及格式
        self.visu_param = dict(
            active_meteors=[
                "draw", {
                    "type": "rectangle",
                    "color": "as-input"
                }
            ],
            active_pts=[
                "draw", {
                    "type": "circle",
                    "position": "as-input",
                    "color": "as-input",
                    "radius": 2,
                    "thickness": -1
                }
            ],
            score_bg=[
                "draw", {
                    "type": "rectangle",
                    "position": "as-input",
                    "color": "as-input",
                    "thickness": -1
                }
            ],
            score_text=["text", {
                "position": "as-input",
                "color": "white"
            }])

    def update(self, cur_frame, lines, cates):
        """
        更新流星序列的主要函数。

        原则上可以在有新响应时更新，实际为了报告效率，可以无流星时每5-10s执行一次。

        Args:
            cur_frame (_type_): _description_
            lines (_type_): _description_
        """
        # 维护活跃流星序列：将已经超过最长时间检测未响应的潜在流星序列移出，将满足条件的流星放入完成序列。
        self.cur_frame = cur_frame
        temp_waiting_meteor, drop_list = [], []
        for ms in self.active_meteor:
            if self.cur_frame - ms.last_activate_frame >= self.max_interval:
                if (self.prob_meteor(ms) > self.det_thre):
                    # 没有后校验的情况下，UNKNOWN，PLANE类型不给予输出
                    if self.met_exporter.recheck or not (ms.cate in [
                            Name2Label.UNKNOWN_AREA, Name2Label.PLANE
                    ]):
                        temp_waiting_meteor.append(ms)
                    else:
                        drop_list.append(ms)
                else:
                    drop_list.append(ms)
        # 维护
        for ms in drop_list:
            self.active_meteor.remove(ms)
            self.ended_meteor.append(self.get_met_attr(ms))
        for ms in temp_waiting_meteor:
            self.active_meteor.remove(ms)
            self.ended_meteor.append(self.get_met_attr(ms))

        # drop的部分不进行合并，直接构建序列
        self.met_exporter.export(self.met_exporter.DROP_FLAG,
                                 [self.get_met_attr(ms) for ms in drop_list])

        self.waiting_meteor.extend(temp_waiting_meteor)

        # 整合待导出序列：如果没有活跃的潜在流星，则导出
        if len(self.waiting_meteor) > 0:
            no_prob_met = True
            for ms in self.active_meteor:
                if self.prob_meteor(ms) > self.det_thre and \
                    (ms.start_frame - self.waiting_meteor[-1].last_activate_frame<= self.max_interval):
                    no_prob_met = False
                    break
            if no_prob_met:
                waiting_meteor = [
                    self.get_met_attr(ms) for ms in self.waiting_meteor
                ]
                # sort meteors in ASC order to avoid time fmt error
                waiting_meteor.sort(key=lambda ms: ms["start_frame"])
                self.met_exporter.export(self.met_exporter.ACTIVE_FALG,
                                         waiting_meteor)
                self.waiting_meteor.clear()

        # 对新的line进行判断
        num_activate = len(self.active_meteor)

        # NMS
        self.line_type, self.lines = [], []
        if len(lines) > 0:
            # TODO: 主要是LineDetector的结果需要；未来考虑优化到LineDetector检测器内部或移除。
            # 目前是利用CATE进行区分: 目前LineDetector统一返回CATE=-1; NMS将流星转换为label=0，非流星为label=-1
            if cates[0] == -1:
                self.line_type, self.lines = lineset_nms(
                    lines, self.thre2, self.drct_prob_func)
            else:
                self.line_type, self.lines = cates, lines

        for cate, line in zip(self.line_type, self.lines):
            # 如果某一序列已经开始，则可能是其中间的一部分。
            # 考虑到基本不存在多个流星交接的情况，如果属于某一个，则直接归入即可。
            # TODO: cur_frame+-eframe fixed!!

            is_in_series = False
            for ms in self.active_meteor[:num_activate]:
                # Area不再接收line性质的更新
                # TODO: Area相关的逻辑需要进一步迭代。目前Area按照-1处置，可能无法召回一些精灵类型的目标
                # Another TODO: 基于模型的方法（和传统方法）的类别标签给出方法可能需要进一步迭代。
                if ms.cate == -1 and cate == 0: continue

                is_in = ms.may_in_series(line, cur_frame)
                if is_in:
                    ms.update(self.cur_frame, line, update_type=cate)
                    is_in_series = True
                    break
            # 如果不属于已存在的序列，则为其构建新的序列开头
            if is_in_series:
                continue

            self.active_meteor.insert(
                len(self.active_meteor) - 1,
                MeteorSeries(max(self.cur_frame - 2 * self.eframe, 0),
                             self.cur_frame,
                             line,
                             max_acceptable_dist=self.thre2,
                             max_acti_frame=self.max_acti_frame,
                             cate=cate))

    def visu(self, frame_num):
        active_meteors, active_pts = [], []
        score_text, score_bg = [], []
        for ms in self.active_meteor:
            pt1, pt2 = ms.range
            color = color_mapper(self.prob_meteor(ms))

            active_meteors.append({"position": (pt1, pt2), "color": color})

            # TODO: 检查是否交互计算时也只计算了最近的点。
            # 只打印最近的响应点
            first = np.where(ms.coord_list.frame_num >= frame_num -
                             self.max_acti_frame)[0]
            first = len(
                ms.coord_list.frame_num) if len(first) == 0 else first[0]
            for pts in ms.coord_list[first:]:
                pt_x, pt_y = pts
                active_pts.append({"position": (pt_x, pt_y), "color": color})

            # print score
            pt1 = [min(pt1[0], pt2[0]), min(pt1[1], pt2[1])]
            score_bg.append({
                "position": (pt1, pt_offset(pt1, (35, -10))),
                "color": color
            })
            score_text.append({
                "position": pt1,
                "text": f"{self.prob_meteor(ms):.2f}"
            })

        return dict(active_meteors=active_meteors,
                    active_pts=active_pts,
                    score_text=score_text,
                    score_bg=score_bg)

    def clear(self):
        """将当前时间更新至无穷久以后，清空列表。
        应当在结束时仅调用一次。

        Raises:
            StopIteration: _description_

        Returns:
            _type_: _description_
        """
        self.update(np.inf, [], [])
        self.met_exporter.export(self.met_exporter.END_FLAG, [])

    def prob_meteor(self, met):
        # 用于估计met实例属于流星序列的概率。
        # 拟借助几个指标
        # 1. 总速度/总长度
        # 2. 平均响应长度（暂未实现）
        # 3. 直线拟合情况（暂未实现）

        # AREA目前按照排异移除掉...或者需要另外给一个头？
        #type_prob = 0 if met.cate == 1 else 1

        # 对短样本实现一定的宽容
        len_prob = self.len_prob_func(met.dist)

        # 排除总时长过长/过短
        time_prob = self.time_prob_func(met.duration)
        # 排除速度过快/过慢
        speed_prob = self.speed_prob_func(met.speed)
        # 计算直线情况
        #print(met.drct_list)
        drct_prob = self.drct_prob_func(met.drst_std)

        return int(time_prob * speed_prob * len_prob * drct_prob * 100) / 100

    def get_met_attr(self, met) -> dict:
        """将met的点集序列转换为属性字典。

        Args:
            met (_type_): _description_

        Returns:
            dict: _description_
        """
        pt1, pt2 = met.sort_range
        dist = np.sqrt(pt_len_xy(pt1, pt2))

        return dict(start_time=self.frame2ts(met.start_frame),
                    start_frame=met.start_frame,
                    end_time=self.frame2ts(met.end_frame),
                    last_activate_frame=met.last_activate_frame,
                    last_activate_time=self.frame2ts(met.last_activate_frame),
                    duration=met.duration,
                    speed=np.round(met.speed, 3),
                    dist=np.round(dist, 3),
                    num_pts=len(met.coord_list),
                    category=ID2NAME[met.cate],
                    pt1=pt1,
                    pt2=pt2,
                    drct_loss=met.drst_std,
                    score=self.prob_meteor(met))

    def frame2ts(self, frame: int) -> str:
        return frame2ts(frame, self.fps)


class MeteorSeries(object):
    """用于整合检测结果，排异和给出置信度的流星序列。

    Args:
        object (_type_): _description_
    """

    def __init__(self, start_frame, cur_frame, init_box, max_acceptable_dist,
                 max_acti_frame, cate):
        self.coord_list = PointList()
        self.drct_list = []
        self.coord_list.extend(self.box2coord(init_box), cur_frame)
        if cate == 0:
            self.drct_list.append(drct_line(init_box))
        self.start_frame = start_frame
        self.end_frame = cur_frame
        self.last_activate_frame = cur_frame
        self.max_acti_frame = max_acti_frame
        self.max_acceptable_dist = max_acceptable_dist
        self.cate = cate
        self.calc_range()

    @property
    def drst_std(self):
        if len(self.drct_list) == 0: return 0
        drct_copy = np.array(self.drct_list.copy())
        std1 = np.std(np.sort(drct_copy)[:-1]) if len(
            drct_copy) >= 3 else np.std(drct_copy)
        drct_copy[drct_copy > np.pi / 2] -= np.pi
        std2 = np.std(np.sort(drct_copy)[:-1]) if len(
            drct_copy) >= 3 else np.std(drct_copy)
        return min(std1, std2)  # type: ignore

    @property
    def duration(self):
        return self.last_activate_frame - self.start_frame + 1

    def calc_range(self):
        """TODO: 这个是高耗时步骤。需要优化。

        Returns:
            _type_: _description_
        """
        self.range = [
            int(min([x[0] for x in self.coord_list])),
            int(min([x[1] for x in self.coord_list]))
        ], [
            int(max([x[0] for x in self.coord_list])),
            int(max([x[1] for x in self.coord_list]))
        ]

    @property
    def sort_range(self):
        """range的增强版，按照时间顺序给出起止点组合
        """
        [x0, y0], [x1, y1] = self.range
        e_x, e_y = self.coord_list[np.argmin(self.coord_list.frame_num)]
        l_x, l_y = self.coord_list[np.argmax(self.coord_list.frame_num)]
        if e_x > l_x:
            x0, x1 = x1, x0
        if e_y > l_y:
            y0, y1 = y1, y0
        return [x0, y0], [x1, y1]

    @property
    def dist(self):
        return pt_len_xy(*self.range)**(1 / 2)

    @property
    def speed(self):
        # TODO: 有个问题：我这个速度是不是应该考虑fps的影响来着...
        return self.dist / (self.end_frame - self.start_frame + 1e-6)

    def box2coord(self, box):
        if len(box) == 4:
            return [box[0], box[1]], [box[2], box[3]], [(box[0] + box[2]) // 2,
                                                        (box[1] + box[3]) // 2]
        else:
            return box

    def update(self, new_frame, new_box, update_type):
        # TODO: 兼容运行。这一块逻辑后续需要优化。
        # 如果label>=1，则是由模型预测的。
        # 飞机线 (=1) 与 红色精灵 (=2) 目前仍按照直线处理。
        if update_type in [0, 1, 2]:
            new_box = [new_box[:2], new_box[2:]]
        (x1, y1), (x2, y2) = self.range
        for pt in new_box:
            if not ((x1 <= pt[0] <= x2) and (y1 <= pt[1] <= y2)):
                self.end_frame = new_frame
                break
        self.last_activate_frame = new_frame
        self.coord_list.extend(new_box, new_frame)
        # range由calc_range更新，除去init外每次仅在update时更新
        self.calc_range()
        if update_type in [0, 1, 2]:
            self.drct_list.append(drct(new_box))

    def may_in_series(self, new_box, cur_frame):
        # 策略一：最后近邻法（对于有尾迹的判断不准确）
        #if pt_len(self.box2coord(new_box)+self.coord_list[-1])<self.max_acceptable_dist:
        #    return True
        # 策略二：近邻法（对于距离中间点近的，采取收入但不作为边界点策略）
        first = np.where(self.coord_list.frame_num >= cur_frame -
                         self.max_acti_frame)[0]
        first = len(self.coord_list.frame_num) if len(first) == 0 else first[0]
        for tgt_pt in self.box2coord(new_box):
            for in_pt in self.coord_list[first:]:
                if pt_len_xy(tgt_pt, in_pt) < self.max_acceptable_dist:
                    return True
        return False


class PointList(object):

    def __init__(self) -> None:
        self.pts = np.zeros((0, 2), dtype=np.int32)
        self.frame_num = np.zeros((0, ), dtype=np.int16)

    def append(self, new_pt, frame):
        if new_pt.shape == (2, ):
            new_pt = new_pt.reshape(-1, 2)
        self.pts = np.concatenate([self.pts, new_pt], axis=0)
        self.frame_num = np.concatenate(
            [self.frame_num, np.array(frame)], axis=0)

    def extend(self, new_pts, frame):
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

    def __repr__(self) -> str:
        return "[" + ",".join(
            ["[" + ",".join(str(p) for p in x) + "]" for x in self.pts]) + "]"

    def __getitem__(self, i):
        return self.pts[i]

    def __len__(self):
        return len(self.pts)


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
    ACTIVE_FALG = "ACTIVE_FLAG"

    def __init__(self, runtime_size, raw_size, recheck_cfg, video_loader,
                 logger, max_interval) -> None:
        self.queue = queue.Queue()
        self.recheck = recheck_cfg.switch
        self.logger = logger
        self.max_interval = max_interval
        if self.recheck:
            self.recheck_loader = video_loader
            self.recheck_model = init_model(recheck_cfg.model)
        # Rescale: 用于将结果放缩回原始分辨率的放缩倍率。
        self.raw_size = raw_size
        self.rescale_ratio = [x / y for x, y in zip(raw_size, runtime_size)]
        self.export_loop = threading.Thread(target=self.loop, daemon=True)
        self.export_loop.start()
        self.save_path = recheck_cfg.save_path
        if self.save_path:
            self.logger.info(
                f"Rechecked imgs will be saved to \"{self.save_path}\".")

    def export(self, flag, data):
        self.queue.put([flag, data])

    def loop(self):
        """
                 (what input)
        met_obj -> met_dict -> output_dict -> output_json

        Raises:
            KeyError: _description_
        """
        flag, data = self.queue.get()
        while flag in [self.ACTIVE_FALG, self.DROP_FLAG]:
            if flag == self.DROP_FLAG:
                for ms_attr in data:
                    met = self.init_output_dict(ms_attr)
                    # 坐标修正和序列化
                    self.logger.dropped(self.cvt2json(met))
            else:
                # ACTIVE_FLAG
                output_dict = dict()
                final_list = []
                for ms_attr in data:
                    if len(output_dict) == 0:
                        output_dict = self.init_output_dict(ms_attr)
                        continue
                    # TODO: 这个max_interval似乎存在复用的歧义性。后续可以考虑独立开来
                    if ms_attr["start_frame"] < output_dict[
                            'end_frame'] + self.max_interval:
                        if ms_attr["last_activate_frame"] > output_dict[
                                "end_frame"]:
                            output_dict["end_frame"] = ms_attr[
                                "last_activate_frame"]
                            output_dict["end_time"] = ms_attr[
                                "last_activate_time"]
                        output_dict["target"].append(ms_attr)
                    else:
                        # 上一片段已经结束（最大间隔超过max_interval）
                        final_list.append(output_dict)
                        output_dict = self.init_output_dict(ms_attr)
                if len(output_dict) != 0:
                    final_list.append(output_dict)
                if self.recheck:
                    final_list = self.recheck_progress(final_list)
                for met in final_list:
                    # 坐标修正和序列化
                    self.logger.meteor(self.cvt2json(met))
            # get next
            flag, data = self.queue.get()
        if flag != self.END_FLAG:
            raise KeyError(
                f"Unexpected flag received. Except [{self.ACTIVE_FALG}"
                f"{self.DROP_FLAG},{self.END_FLAG}], got {flag} instead.")

    def cvt2json(self, meteor_dict: dict) -> str:
        """将流星列表转化为JSON-string列表.

        Args:
            meteor_list (_type_): _description_
        """
        # 转换尺寸，将输出的所有尺寸放缩回真实位置
        for meteor in meteor_dict["target"]:
            meteor["pt1"] = scale_to(meteor["pt1"], self.rescale_ratio)
            meteor["pt2"] = scale_to(meteor["pt2"], self.rescale_ratio)
        return json.dumps(meteor_dict)

    def init_output_dict(self, attributes):
        return dict(start_frame=attributes['start_frame'],
                    start_time=attributes['start_time'],
                    end_time=attributes['last_activate_time'],
                    end_frame=attributes['last_activate_frame'],
                    video_size=self.raw_size,
                    target=[attributes])

    def recheck_progress(self, final_list):
        """_summary_

        Args:
            final_list (_type_): 包含若干个整片段
        """
        # 片段级重校验
        # TODO: 存在潜在的可能性，删除中间片段之后前后间隔过长。
        # 重构Collector时修复。
        new_final_list = []
        for output_dict in final_list:
            stacked_img = max_stacker(video_loader=self.recheck_loader,
                                      start_frame=output_dict["start_frame"],
                                      end_frame=output_dict["end_frame"])
            bbox_list, cls_list = self.recheck_model.forward(stacked_img)
            # 匹配bbox，丢弃未检出box，修改与类别得分为模型预测得分
            raw_bbox_list = [[*x["pt1"], *x["pt2"]]
                             for x in output_dict["target"]]
            matched_pairs = box_matching(bbox_list, raw_bbox_list)
            fixed_output_dict = dict(video_size=output_dict["video_size"],
                                     target=[])
            for l, r in matched_pairs:
                # if cate->1, drop.
                if cls_list[l] == 1:
                    continue
                sure_meteor = output_dict["target"][r]
                sure_meteor["category"] = ID2NAME.get(cls_list[l], "UNDEFINED")
                # TODO: score没有带出来
                sure_meteor["score"] = 1.0
                fixed_output_dict["target"].append(sure_meteor)
            # after fix. to be optimized.
            if len(fixed_output_dict["target"]) == 0:
                continue
            # 为fixed_output_dict重新计算准确的起止时间
            fixed_output_dict["start_time"] = min(
                [x["start_time"] for x in fixed_output_dict["target"]])
            fixed_output_dict["end_frame"] = max([
                x["last_activate_frame"] for x in fixed_output_dict["target"]
            ])
            fixed_output_dict["end_time"] = max(
                [x["last_activate_time"] for x in fixed_output_dict["target"]])
            new_final_list.append(fixed_output_dict)
            # 用于观察效果的临时打点。后续可以转为进行时截图生成
            if self.save_path:
                save_img(
                    stacked_img,
                    f"{self.save_path}/test_{fixed_output_dict['start_time'].replace(':','.')}_{fixed_output_dict['end_time'].replace(':','.')}.jpg",
                    quality=80,
                    compressing=0)
        return new_final_list