from MetLib.VideoLoader import ThreadVideoReader
from MetLib.utils import preprocessing, load_video_and_mask
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import tqdm

video_name = "F:\\补充测试片段\\2022-10-26 19-51-38.mp4"
mask_name = "F:\\补充测试片段\\2022-10-26 19-51-38-mask.jpg"
resize_param = [960, 540]
n_frames = 200
video, mask = load_video_and_mask(video_name, mask_name, resize_param)
try:
    video_loader = ThreadVideoReader(
        video, n_frames,
        partial(preprocessing, mask=mask, resize_param=resize_param))
    video_loader.start()
    f_sum = np.zeros((n_frames, *reversed(resize_param)), dtype=np.float)
    for i in tqdm.tqdm(range(n_frames)):
        f_sum[i] = video_loader.pop(1)[0]

finally:
    video_loader.stop()

    f_avg = np.mean(f_sum, axis=0)

    diff = (f_sum - f_avg)
    diff = np.reshape(diff, (-1, ))
    print(max(diff), min(diff))
    std_err = np.std(diff) * 3
    print(len(np.where(np.abs(diff) > std_err)[0]))
    plt.hist(diff, bins=200, range=(-20, 20))
    plt.show()
