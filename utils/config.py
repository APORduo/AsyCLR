
import os
os.sys.path.append(os.getcwd())
from yacs.config import CfgNode
import json
import yaml
import argparse
import pdb
def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param

def load_yml(path):
    with open(path,'r')as fp:
        param = yaml.safe_load(fp)
    return param
cfg = CfgNode(load_json('configs/base.json'),new_allowed=True)

# opt_cifar = cfg.clone()
# opt_cifar.merge_from_file('configs/cifar.yml')
# opt_cub = cfg.clone()
# opt_cub.merge_from_file('configs/cub.yml')
opt_mini =  cfg.clone()
opt_mini.merge_from_file('configs/mini.yml')


parser = argparse.ArgumentParser("Base config file")
parser.add_argument("--config", type= str,default=None)

args,unknow = parser.parse_known_args()
cfg.merge_from_file(args.config) if args.config else None
cfg.config = args.config





