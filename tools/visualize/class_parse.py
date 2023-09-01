import argparse
import os
import os.path as osp
import time
import warnings
import sys
sys.path.append('/home/liu/ytx/SS-OD')
from mmcv import Config, DictAction
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmssod.utils.patch import patch_config
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.image import tensor2imgs
from mmdet.core.visualization import imshow_gt_det_bboxes
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import numpy as np
from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmssod.utils.cluster import K_means
import matplotlib.pylab as plt
import seaborn as sns
import pickle
def read_pesudo_list(file):
    with open(file,'rb') as f:
        data =  pickle.load(f)
    return data

CLASSES = np.array(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor', 'BG'])



cfg_path = '/home/liu/ytx/SS-OD/outputs/NegativeLoss/9k_rcnn_voc_neg_unreliable_0.3_thr_0.00001_ignore_false/rcnn_voc_neg_thr_0.00001_ignore_false.py'
check_point = '/home/liu/ytx/SS-OD/outputs/NegativeLoss/9k_rcnn_voc_neg_unreliable_0.3_thr_0.00001_ignore_false/epoch_24.pth'
outputs_dir = '/home/liu/ytx/SS-OD/outputs/pesudo_infos/'
region_path = outputs_dir + 'error_info_75_neg_unreliable_0.5_thr_0.0005_unreliable_con_without_unrelia.pkl'


def main():

    cfg = Config.fromfile(cfg_path)

    cfg.model.train_cfg = None
    if cfg.get('train_cfg') != None:
        cfg.model = cfg.model_wrapper
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, check_point, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = CLASSES
    print(model.projector_pos_queue.shape)
    print(model.projector_neg_queue.shape)
    cls_feats = []
    cls_feats_t = []
    for cls_idx in range(20):
        cls_feats.append(model.projector_pos_queue[cls_idx])
        cls_feats_t.append(model.projector_pos_queue[cls_idx].t())

    print(cls_feats[0].shape)
    print(cls_feats_t[0].shape)

    corr_map = torch.zeros((21, 21))
    for i in range(20):
        # corr_scores = np.zeros((20, 20))
        for j in range(20):
            corr = torch.mm(cls_feats_t[i], cls_feats[j]).max()
            if i == j:
                corr_map[i, j] = 0.0
            else:
                corr_map[i, j] = corr
            # print(corr.shape)
            # print(corr)
        bg_corr = torch.mm(cls_feats_t[i], model.projector_neg_queue).max()
        corr_map[i, 20] = bg_corr
    print(corr_map)

    f, ax = plt.subplots(figsize=(12, 9))
    # ax = sns.heatmap((1. - corr_map).numpy() , vmin=0.3)
    ax = sns.heatmap(corr_map.numpy(),
                     # vmin=0.60, vmax=0.8,
                     vmin=0.99, vmax=1.0,
                     cmap=sns.light_palette("purple", n_colors=5),
                     xticklabels=CLASSES,
                     yticklabels=CLASSES)
    plt.show()
    # sns.heatmap()

    # region_info = read_pesudo_list(region_path)
    # print(region_info[:, :])
    # f, ax = plt.subplots(figsize=(12, 9))
    # ax = sns.heatmap(region_info,
    #                  vmin=0, vmax=80,
    #                  cmap=sns.light_palette("purple", n_colors=4),
    #                  xticklabels=CLASSES,
    #                  yticklabels=CLASSES)
    # plt.show()

if __name__ == '__main__':
    main()