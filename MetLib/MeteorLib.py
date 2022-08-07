import numpy as np
import datetime
import json
import cv2

pt_len_4 = lambda pts: (pts[3] - pts[1])**2 + (pts[2] - pts[0])**2
pt_len_xy = lambda pt1, pt2: (pt1[1] - pt2[1])**2 + (pt1[0] - pt2[0])**2
color_map = [[0,0,255],[0,255,0]]

class MeteorCollector(object):
    """
    全局的流星统计模块。用于记录和管理所有的响应，整合成正在发生（或已经结束）的检测序列。
    """

    def __init__(self, min_len, max_interval, det_thre, time_range,
                 speed_range, thre2, fps) -> None:
        self.min_len = min_len
        self.max_interval = max_interval
        self.det_thre = det_thre
        self.active_meteor = [
            MeteorSeries(np.inf, [-100, -100, -100, -100], (-np.nan, -np.nan),
                         (-np.nan, -np.nan), np.nan, np.nan)
        ]
        self.waiting_meteor = []
        self.cur_frame = 0
        self.thre2 = thre2
        self.time_range = time_range
        self.speed_range = speed_range
        self.fps = fps

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
        met_list= []
        for ms in self.active_meteor:
            if self.cur_frame - ms.last_activate_frame > self.max_interval:
                if ms.prob_meteor() >= self.det_thre:
                    temp_waiting_meteor.append(ms)
                else:
                    drop_list.append(ms)
                    pass
        # 维护
        for ms in drop_list:
            self.active_meteor.remove(ms)
        for ms in temp_waiting_meteor:
            self.active_meteor.remove(ms)
        self.waiting_meteor.extend(temp_waiting_meteor)
        # 整合待导出序列：如果没有活跃的潜在流星，则导出
        if len(self.waiting_meteor) > 0:
            no_prob_met = True
            for ms in self.active_meteor:
                if ms.prob_meteor() >= self.det_thre:
                    no_prob_met = False
                    break
            if no_prob_met:
                met_list = self.jsonize_waiting_meteor()
                self.waiting_meteor.clear()

        # 对新的line进行判断
        num_activate = len(self.active_meteor)
        for line in lines:
            # 如果某一序列已经开始，则可能是其中间的一部分。
            # 考虑到基本不存在多个流星交接的情况，如果属于某一个，则直接归入即可。
            is_in_series = False
            for ms in self.active_meteor[:num_activate]:
                is_in = ms.may_in_series(line)
                if is_in:
                    ms.update(self.cur_frame, line)
                    is_in_series = True
                    break
            # 如果不属于已存在的序列，并且长度满足触发阈值，则为其构建新的序列开头
            if is_in_series or pt_len_4(line) < self.min_len:
                continue
            self.active_meteor.insert(
                len(self.active_meteor) - 1,
                MeteorSeries(
                    self.cur_frame,
                    line,
                    time_range=self.time_range,
                    speed_range=self.speed_range,
                    max_acceptable_dist=self.thre2,
                    fps=self.fps))
        return met_list, drop_list

    def jsonize_waiting_meteor(self):
        def init_output_dict(ms, ms_json):
            return dict(
                start_time=ms_json['start_time'],
                end_time=ms_json['end_time'],
                end_frame=ms.end_frame,
                target=[ms_json])

        output_dict = dict()
        final_list = []
        for ms in self.waiting_meteor:
            ms_json = ms.property_json
            if len(output_dict) == 0:
                output_dict = init_output_dict(ms, ms_json)
                continue
            if ms.start_frame < output_dict['end_frame'] + self.max_interval:
                output_dict.update(
                    end_time=ms_json['end_time'], end_frame=ms.end_frame)
                output_dict["target"].append(ms_json)
            else:
                final_list.append(output_dict)
                output_dict = init_output_dict(ms, ms_json)
        if len(output_dict) != 0:
            final_list.append(output_dict)
        final_list = [json.dumps(x) for x in final_list]
        return final_list

    def draw_on_img(self, img):
        #raise NotImplementedError("Global vars are not solved until next update.")
        draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for ms in self.active_meteor:
            pt1, pt2 = ms.range
            draw_img = cv2.rectangle(draw_img, pt1, pt2, color_map[ms.prob_meteor()], 2)
        return draw_img


class MeteorSeries(object):
    """用于整合检测结果，排异和给出置信度的流星序列。

    Args:
        object (_type_): _description_
    """

    def __init__(self, frame, init_box, time_range, speed_range,
                 max_acceptable_dist, fps):
        self.coord_list = []
        self.len_list = []
        self.coord_list.append(self.box2coord(init_box))
        self.len_list.append(pt_len_4(init_box))
        self.start_frame = frame
        self.end_frame = frame
        self.last_activate_frame = frame
        self.max_acceptable_dist = max_acceptable_dist
        self.time_range = time_range
        self.speed_range = speed_range
        self.fps = fps

    def __repr__(self) -> str:
        return "Duration %s frames; (Dist=%s); speed=%.2f px(s)/frame; \"%s - %s : %s - %s\"" % (
            self.duration, self.dist, self.speed, self.start_frame,
            self.last_activate_frame, self.range[0], self.range[1])

    @property
    def property_json(self) -> dict:
        return dict(
            start_time=self.frame2ts(self.start_frame),
            end_time=self.frame2ts(self.end_frame),
            last_activate_time=self.frame2ts(self.last_activate_frame),
            duration=self.duration,
            speed=self.speed,
            dist=self.dist,
            pt1=self.range[0],
            pt2=self.range[1])

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
        return self.dist / (self.end_frame - self.start_frame + 1e-6)

    def frame2ts(self, frame):
        return datetime.datetime.strftime(
            datetime.datetime.utcfromtimestamp(frame / self.fps),
            "%H:%M:%S.%f")

    def box2coord(cls, box):
        return [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]

    def update(self, new_frame, new_box):
        pt1, pt2 = new_box[:2], new_box[2:]
        (x1, y1), (x2, y2) = self.range
        if not (((x1 <= pt1[0] <= x2) and (y1 <= pt1[1] <= y2)) and
                ((x1 <= pt2[0] <= x2) and (y1 <= pt2[1] <= y2))):
            self.end_frame = new_frame
        self.last_activate_frame = new_frame
        self.coord_list.extend([pt1, pt2])
        self.len_list.append(pt_len_4(new_box))

    def may_in_series(self, new_box):
        # 策略一：最后近邻法（对于有尾迹的判断不准确）
        #if pt_len(self.box2coord(new_box)+self.coord_list[-1])<self.max_acceptable_dist:
        #    return True
        # 策略二：近邻法（对于距离中间点近的，采取收入但不作为边界点策略）
        for in_pt in self.coord_list:
            if pt_len_xy(self.box2coord(new_box),
                         in_pt) < self.max_acceptable_dist:
                return True
        return False

    def is_in_range(self, value, range_tuple):
        if range_tuple[0] <= value <= range_tuple[1]:
            return True
        return False

    def prob_meteor(self):
        # 自身为流星序列的概率。
        # 拟借助几个指标
        # 1. 总速度/总长度
        # 2. 平均响应长度（暂未实现）
        # 3. 直线拟合情况（暂未实现）

        # 排除总时长过长/过短
        if not self.is_in_range(self.duration, self.time_range):
            return 0
        # 排除速度过快/过慢
        if not self.is_in_range(self.speed, self.speed_range):
            return 0

        return 1