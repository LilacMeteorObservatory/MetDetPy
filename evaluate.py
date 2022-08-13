import argparse
import json

from core import detect_video
from MetLib.utils import Munch

parser = argparse.ArgumentParser(description='MetDetPy Evaluater.')

parser.add_argument(
    '--videos',
    help="json file of test videos.",
    default="./test_video.json")
parser.add_argument(
    '--cfg', '-C', help="Config file.", default="./config.json")

parser.add_argument(
        '--debug',
        '-D',
        action='store_true',
        help="Apply Debug Mode",
        default=False)

args = parser.parse_args()

## Load json files.

with open(args.videos, mode='r', encoding='utf-8') as f:
    video_dict = json.load(f)

with open('config.json', mode='r', encoding='utf-8') as f:
    cfg = Munch(json.load(f))

# single_test
#test(*fp_database[-2], cfg, True)

tasks = []
for obj in video_dict["true_positive"]:
    obj=Munch(obj)
    detect_video(obj.video,obj.mask,cfg,args.debug)


# overall_test
#:
#    test(*item, cfg, False)
