import json

import cv2
import numpy as np

from .utils import frame2ts, color_interpolater, pt_len_xy, lineset_nms, drct

color_mapper = color_interpolater([[128, 128, 128], [128, 128, 128],
                                   [0, 255, 0]])


def scale_to(pt, rescale):
    return [int(x * y) for x, y in zip(pt, rescale)]


def init_output_dict(attributes, size):
    return dict(start_time=attributes['start_time'],
                end_time=attributes['last_activate_time'],
                end_frame=attributes['last_activate_frame'],
                video_size=size,
                target=[attributes])


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
    全局的流星统计模块。用于记录和管理所有的响应，整合成正在发生（或已经结束）的检测序列。
    """

    def __init__(self, min_len, max_interval, det_thre, time_range,
                 speed_range, thre2, eframe, drct_range, fps, runtime_size,
                 raw_size) -> None:
        self.min_len = min_len
        self.max_interval = max_interval
        self.max_acti_frame = max_interval
        self.det_thre = det_thre
        self.active_meteor = [
            MeteorSeries(np.inf, np.inf, [-100, -100, -101, -101], np.nan,
                         np.nan, "None")
        ]
        self.waiting_meteor = []
        self.ended_meteor = []
        self.cur_frame = 0
        self.thre2 = thre2
        self.speed_range = speed_range
        self.eframe = eframe
        self.fps = fps
        self.raw_size = raw_size
        # Rescale: 用于将结果放缩回原始分辨率的放缩倍率。
        self.rescale_ratio = [x / y for x, y in zip(raw_size, runtime_size)]
        # 调整time的验证下界
        time_range[0] = max(time_range[0], int(4 * self.eframe + 2))
        self.time_prob_func = create_prob_func(time_range)
        self.speed_prob_func = create_prob_func(speed_range)
        self.len_prob_func = create_prob_func((min_len, np.inf))
        self.drct_prob_func = create_prob_func(drct_range)

    def update(self, cur_frame, lines):
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
        met_list = []
        for ms in self.active_meteor:
            if self.cur_frame - ms.last_activate_frame >= self.max_interval:
                if self.prob_meteor(ms) > self.det_thre:
                    temp_waiting_meteor.append(ms)
                else:
                    drop_list.append(ms)
        # 维护
        for ms in drop_list:
            self.active_meteor.remove(ms)
            self.ended_meteor.append(self.get_met_attr(ms))
        for ms in temp_waiting_meteor:
            self.active_meteor.remove(ms)
            self.ended_meteor.append(self.get_met_attr(ms))

        # drop的部分不进行合并，直接构建序列化
        drop_list = self.list2json([
            init_output_dict(self.get_met_attr(ms), size=self.raw_size)
            for ms in drop_list
        ])
        self.waiting_meteor.extend(temp_waiting_meteor)

        # 整合待导出序列：如果没有活跃的潜在流星，则导出
        if len(self.waiting_meteor) > 0:
            no_prob_met = True
            for ms in self.active_meteor:
                if self.prob_meteor(ms) >= self.det_thre and \
                    (ms.start_frame - self.waiting_meteor[-1].last_activate_frame<= self.max_interval):
                    no_prob_met = False
                    break
            if no_prob_met:
                met_list = self.jsonize_waiting_meteor()
                self.waiting_meteor.clear()

        # 对新的line进行判断
        num_activate = len(self.active_meteor)
        drcts = []
        # 做NMS
        if len(lines) > 0:
            drcts, lines = lineset_nms(lines, self.thre2)
        self.drcts = drcts
        self.lines = lines

        for line_drct, line in zip(drcts, lines):
            # 如果某一序列已经开始，则可能是其中间的一部分。
            # 考虑到基本不存在多个流星交接的情况，如果属于某一个，则直接归入即可。
            # TODO: cur_frame+-eframe fixed!!
            
            met_type = "area" if self.drct_prob_func(line_drct) < 0.75 else "line"
            is_in_series = False
            for ms in self.active_meteor[:num_activate]:
                # Area不再接收line性质的更新
                if ms.met_type=="area" and met_type=="line" : continue
                
                is_in = ms.may_in_series(line, cur_frame)
                if is_in:
                    ms.update(self.cur_frame, line, update_type = met_type)
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
                             met_type=met_type))
        return met_list, drop_list

    def list2json(self, meteor_list):
        """将流星列表转化为JSON-string列表.

        Args:
            meteor_list (_type_): _description_
        """
        # 转换尺寸，将输出的所有尺寸放缩回真实位置
        for line in meteor_list:
            for meteor in line["target"]:
                meteor["pt1"] = scale_to(meteor["pt1"], self.rescale_ratio)
                meteor["pt2"] = scale_to(meteor["pt2"], self.rescale_ratio)
        return [json.dumps(x) for x in meteor_list]

    def jsonize_waiting_meteor(self):
        output_dict = dict()
        final_list = []
        for ms in self.waiting_meteor:
            ms_attrbutes = self.get_met_attr(ms)
            if len(output_dict) == 0:
                output_dict = init_output_dict(ms_attrbutes,
                                               size=self.raw_size)
                continue
            if ms.start_frame < output_dict['end_frame'] + self.max_interval:
                output_dict.update(end_time=ms_attrbutes['end_time'],
                                   end_frame=ms.end_frame)
                output_dict["target"].append(ms_attrbutes)
            else:
                final_list.append(output_dict)
                output_dict = init_output_dict(ms_attrbutes,
                                               size=self.raw_size)
        if len(output_dict) != 0:
            final_list.append(output_dict)

        # 整理完列表后对其进行坐标修正和序列化
        return self.list2json(final_list)

    def draw_on_img(self, draw_img, frame_num):
        # add timestamp
        h, w, _ = draw_img.shape
        max_len = max(h, w)
        draw_img = cv2.putText(draw_img, self.frame2ts(frame_num),
                               (int(w * 0.01), int(h * 0.98)),
                               cv2.FONT_HERSHEY_COMPLEX, max_len / 1920,
                               (255, 255, 255), 1)
        for line_drct, line in zip(self.drcts,self.lines):
            line_color = [0,0,0] if self.drct_prob_func(line_drct)<0.75 else [0,255,0]
            draw_img = cv2.line(draw_img,
                                line[:2],
                                line[2:],
                                color=line_color,
                                thickness=3)
        for ms in self.active_meteor:
            pt1, pt2 = ms.range
            color = color_mapper(self.prob_meteor(ms))
            first = np.where(
                ms.coord_list.frame_num >= frame_num - self.max_acti_frame)[0]
            first = len(
                ms.coord_list.frame_num) if len(first) == 0 else first[0]
            draw_img = cv2.rectangle(draw_img, pt1, pt2, color, 2)
            for pts in ms.coord_list[first:]:
                pt_x, pt_y = pts
                draw_img = cv2.circle(draw_img, (pt_x, pt_y), 2, color, -1)
            # print score
            pt1 = [min(pt1[0], pt2[0]), min(pt1[1], pt2[1])]
            draw_img = cv2.rectangle(draw_img, pt1, (pt1[0] + 35, pt1[1] - 10),
                                     color, -1)
            draw_img = cv2.putText(draw_img, f"{self.prob_meteor(ms):.2f}",
                                   pt1, cv2.FONT_HERSHEY_COMPLEX,
                                   max_len / 1920, (255, 255, 255), 2)

        return draw_img

    def clear(self):
        """将当前时间更新至无穷久以后，清空列表。
        应当在结束时仅调用一次。

        Raises:
            StopIteration: _description_

        Returns:
            _type_: _description_
        """
        return self.update(np.inf, [])

    def prob_meteor(self, met):
        # 用于估计met实例属于流星序列的概率。
        # 拟借助几个指标
        # 1. 总速度/总长度
        # 2. 平均响应长度（暂未实现）
        # 3. 直线拟合情况（暂未实现）

        # AREA目前按照排异移除掉...或者需要另外给一个头？
        type_prob = 0 if met.met_type=="area" else 1
        
        # 对短样本实现一定的宽容
        len_prob = self.len_prob_func(met.dist)

        # 排除总时长过长/过短
        time_prob = self.time_prob_func(met.duration)
        # 排除速度过快/过慢
        speed_prob = self.speed_prob_func(met.speed)
        # 计算直线情况
        #print(met.drct_list)
        drct_prob = self.drct_prob_func(met.drst_std)

        return int(type_prob * time_prob * speed_prob * len_prob * drct_prob * 100) / 100

    def get_met_attr(self, met) -> dict:
        """将met的点集序列转换为属性字典。

        Args:
            met (_type_): _description_

        Returns:
            dict: _description_
        """
        pt1, pt2 = met.range
        dist = np.sqrt(pt_len_xy(pt1, pt2))

        return dict(start_time=self.frame2ts(met.start_frame),
                    end_time=self.frame2ts(met.end_frame),
                    last_activate_frame=met.last_activate_frame,
                    last_activate_time=self.frame2ts(met.last_activate_frame),
                    duration=met.duration,
                    speed=np.round(met.speed, 3),
                    dist=np.round(dist, 3),
                    num_pts=len(met.coord_list),
                    pt1=pt1,
                    pt2=pt2,
                    drct_loss=met.drst_std,
                    score=self.prob_meteor(met))

    def frame2ts(self, frame):
        return frame2ts(frame, self.fps)


class MeteorSeries(object):
    """用于整合检测结果，排异和给出置信度的流星序列。

    Args:
        object (_type_): _description_
    """

    def __init__(self, start_frame, cur_frame, init_box, max_acceptable_dist,
                 max_acti_frame, met_type):
        self.coord_list = PointList()
        self.drct_list = []
        self.coord_list.extend(self.box2coord(init_box), cur_frame)
        self.drct_list.append(drct(init_box))
        self.start_frame = start_frame
        self.end_frame = cur_frame
        self.last_activate_frame = cur_frame
        self.max_acti_frame = max_acti_frame
        self.max_acceptable_dist = max_acceptable_dist
        self.met_type = met_type

    @property
    def drst_std(self):
        drct_copy = np.array(self.drct_list.copy())
        std1 = np.std(np.sort(drct_copy)
                      [:-1]) if len(drct_copy) >= 3 else np.std(drct_copy)
        drct_copy[drct_copy > np.pi / 2] -= np.pi
        std2 = np.std(np.sort(drct_copy)
                      [:-1]) if len(drct_copy) >= 3 else np.std(drct_copy)
        return min(std1, std2)

    @property
    def duration(self):
        return self.last_activate_frame - self.start_frame + 1

    @property
    def range(self):
        return [
            int(min([x[0] for x in self.coord_list])),
            int(min([x[1] for x in self.coord_list]))
        ], [
            int(max([x[0] for x in self.coord_list])),
            int(max([x[1] for x in self.coord_list]))
        ]

    @property
    def dist(self):
        return pt_len_xy(*self.range)**(1 / 2)

    @property
    def speed(self):
        # TODO: 有个问题：我这个速度是不是应该考虑fps的影响来着...
        return self.dist / (self.end_frame - self.start_frame + 1e-6)

    def box2coord(cls, box):
        return [box[0], box[1]], [box[2], box[3]], [(box[0] + box[2]) // 2,
                                                    (box[1] + box[3]) // 2]

    def update(self, new_frame, new_box, update_type):
        pt1, pt2 = new_box[:2], new_box[2:]
        (x1, y1), (x2, y2) = self.range
        if not (((x1 <= pt1[0] <= x2) and (y1 <= pt1[1] <= y2)) and
                ((x1 <= pt2[0] <= x2) and (y1 <= pt2[1] <= y2))):
            self.end_frame = new_frame
        self.last_activate_frame = new_frame
        self.coord_list.extend([pt1, pt2], new_frame)
        if update_type=="line":
            self.drct_list.append(drct(new_box))

    def may_in_series(self, new_box, cur_frame):
        # 策略一：最后近邻法（对于有尾迹的判断不准确）
        #if pt_len(self.box2coord(new_box)+self.coord_list[-1])<self.max_acceptable_dist:
        #    return True
        # 策略二：近邻法（对于距离中间点近的，采取收入但不作为边界点策略）
        first = np.where(
            self.coord_list.frame_num >= cur_frame - self.max_acti_frame)[0]
        first = len(self.coord_list.frame_num) if len(first) == 0 else first[0]
        for tgt_pt in self.box2coord(new_box):
            for in_pt in self.coord_list[first:]:
                if pt_len_xy(tgt_pt, in_pt) < self.max_acceptable_dist:
                    return True
        return False

    def is_in_range(self, value, range_tuple):
        if range_tuple[0] <= value <= range_tuple[1]:
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