import numpy as np
from multiprocessing import Pool

import torch
from mmdet.core.evaluation.mean_ap import tpfp_default
from mmdet.core.bbox.iou_calculators import build_iou_calculator

def get_cls_results(pesudo_results, gt_results, cls_id):
    cls_dets = []
    cls_gts = []
    cls_gts_ignore = []
    cls_gts_num = 0
    for det in pesudo_results:
        det_inds = det["labels"] == cls_id
        cls_dets.append(det["bboxes"][det_inds, :])

    for gt in gt_results:
        gt_inds = gt["labels"] == cls_id
        cls_gts.append(gt["bboxes"][gt_inds, :])
        cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))
        cls_gts_num = cls_gts_num + np.sum(gt_inds)

    return cls_dets, cls_gts, cls_gts_ignore, cls_gts_num

def cal_recall_precisions(det_bboxes, gts_bboxes, iou_thrs, n_cls):
    num_imgs = len(det_bboxes)
    pool = Pool(4)
    summary = []
    for iou in iou_thrs:
        num_tp, num_fp, num_gts = 0, 0, 0
        for i in range(n_cls):
            cls_dets, cls_gts, cls_gts_ignore, cls_gts_num = get_cls_results(det_bboxes,
                                                                             gts_bboxes, i)
            # print(cls_dets)
            tpfp_fn = tpfp_default
            if not callable(tpfp_fn):
                raise ValueError(
                    f'tpfp_fn has to be a function or None, but got {tpfp_fn}')
            tpfp = pool.starmap(
                tpfp_fn,
                zip(cls_dets, cls_gts, cls_gts_ignore,
                    [iou for _ in range(num_imgs)]))
            if len(tpfp) == 0:
                continue
            # print(tpfp)
            tp, fp = tuple(zip(*tpfp))
            tp = np.sum(np.hstack(tp))
            fp = np.sum(np.hstack(fp))
            num_tp = num_tp + tp
            num_fp = num_fp + fp
            num_gts = num_gts + cls_gts_num
        summary.append({"tp":num_tp, "fp":num_fp, "gts":num_gts})
    return summary


iou_calculator = build_iou_calculator(dict(type='BboxOverlaps2D'))
def cal_unsup_sampling_overlaps(pos_bboxes, neg_bboxes, gts_bboxes):

    pos_bboxes = pos_bboxes.detach()#.cpu()
    neg_bboxes = neg_bboxes.detach()#.cpu()
    gts_bboxes = gts_bboxes.detach()#.cpu()

    if(len(pos_bboxes) == 0):
        pos_overlaps = torch.tensor([]).to(pos_bboxes.device)
    else:
        pos_iou_results = iou_calculator(gts_bboxes, pos_bboxes)
        pos_overlaps, _ = pos_iou_results.max(dim=0)

    if(len(neg_bboxes) == 0):
        neg_overlaps = torch.tensor([]).to(neg_bboxes.device)
    else:
        neg_iou_results = iou_calculator(gts_bboxes, neg_bboxes)
        neg_overlaps, _ = neg_iou_results.max(dim=0)

    return pos_overlaps, neg_overlaps

def cal_bboxes_overlaps(det_bboxes, gts_bboxes, mode='iou'):
    """
    :param det_bboxes: (N1 x 4)
    :param gts_bboxes: (N2 x 4)
    :param mode:
    :return: (N1)
    """
    gts_bboxes = gts_bboxes.detach()
    det_bboxes = det_bboxes.detach()

    if len(det_bboxes) == 0:
        return torch.tensor([]).to(det_bboxes.device), torch.tensor([]).to(det_bboxes.device)

    if len(gts_bboxes) == 0:
        return torch.zeros(len(det_bboxes)).to(det_bboxes.device), torch.zeros(len(det_bboxes)).to(det_bboxes.device).long()

    iou_results = iou_calculator(gts_bboxes, det_bboxes, mode=mode)
    overlaps, inds = iou_results.max(dim=0)

    return overlaps, inds

def cal_bboxes_all_overlaps(det_bboxes, gts_bboxes, mode='iou'):
    """

    :param det_bboxes: (N1 x 4)
    :param gts_bboxes: (N2 x 4)
    :param mode:
    :return: (N1 x N2)
    """
    gts_bboxes = gts_bboxes.detach()
    det_bboxes = det_bboxes.detach()

    # if len(det_bboxes) == 0:
    #     return torch.tensor([]).to(det_bboxes.device)
    #
    # if len(gts_bboxes) == 0:
    #     return torch.zeros(len(det_bboxes)).to(det_bboxes.device)

    iou_results = iou_calculator(det_bboxes, gts_bboxes, mode=mode)
    return iou_results
