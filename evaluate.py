from core import test
import json
import argparse

parser = argparse.ArgumentParser(description='MetDetPy Evaluater.')

parser.add_argument(
    '--videos',
    help="json file of test videos.",
    default="./test_video.json")
parser.add_argument(
    '--cfg', '-C', help="Config file.", default="./config.json")

parser.add_argument(
    '--debug-mode', '-D', help="Apply Debug Mode.", default=False)

args = parser.parse_args()

## Load json files.

with open(args.videos, mode='r', encoding='utf-8') as f:
    video_dict = json.load(f)

with open('config.json', mode='r', encoding='utf-8') as f:
    cfg = json.load(f)

# single_test
#test(*fp_database[-2], cfg, True)

for video,mask in video_dict["false_positive"]:
    test(video,mask,cfg,args.debug_mode)

# overall_test
#:
#    test(*item, cfg, False)
