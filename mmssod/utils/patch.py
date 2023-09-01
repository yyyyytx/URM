import glob
import os
import os.path as osp
import shutil
import types

from mmcv.runner import BaseRunner, EpochBasedRunner, IterBasedRunner
from mmcv.utils import Config

from .vars import resolve

def setup_env(cfg):
    os.environ["WORK_DIR"] = cfg.work_dir


def patch_config(cfg):

    cfg_dict = super(Config, cfg).__getattribute__("_cfg_dict").to_dict()
    cfg_dict["cfg_name"] = osp.splitext(osp.basename(cfg.filename))[0]
    cfg_dict = resolve(cfg_dict)
    cfg = Config(cfg_dict, filename=cfg.filename)

    # enable environment variables
    # setup_env(cfg)
    return cfg

def find_latest_checkpoint(path, ext="pth"):
    if not osp.exists(path):
        return None
    if osp.exists(osp.join(path, f"latest.{ext}")):
        return osp.join(path, f"latest.{ext}")

    checkpoints = glob.glob(osp.join(path, f"*.{ext}"))
    if len(checkpoints) == 0:
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split("_")[-1].split(".")[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path