import os
import pickle
from mmssod.models.utils.eval_utils import cal_recall_precisions, get_cls_results, cal_bboxes_overlaps
import numpy as np

from multiprocessing import Pool

import torch
from mmdet.core.evaluation.mean_ap import tpfp_default

eps = np.finfo(np.float32).eps
iou_thrs = [0.5, 0.75]
def read_pesudo_list(file):
    with open(file,'rb') as f:
        data =  pickle.load(f)
    return data


def write_pesudo_list(pesudo_list, file):
    with open(file,'wb') as f:
        pickle.dump(pesudo_list, f)
    # return data
pesudo_summary_list = []
def cal_summary(det_bboxes_list, gts_bboxes_list):
    pesudo_summary = cal_recall_precisions(det_bboxes_list,
                                           gts_bboxes_list,
                                           iou_thrs,
                                           n_cls=20)
    log_str=""
    for i in range(len(iou_thrs)):
        recalls = pesudo_summary[i]["tp"] / np.maximum(pesudo_summary[i]["gts"], eps)
        precisions = pesudo_summary[i]["tp"] / np.maximum((pesudo_summary[i]["tp"] +
                                                                       pesudo_summary[i]["fp"]), eps)
        log_str += "[Pesudo BBox Summary(Iter)] iou %.2f recall %.4f precisions %.4f tp %d fp %d gts %d \n" % \
                   (iou_thrs[i], recalls, precisions, pesudo_summary[i]["tp"],
                    pesudo_summary[i]["fp"], pesudo_summary[i]["gts"])
    print(pesudo_summary)
    pesudo_summary_list.append(pesudo_summary)
    print(log_str)

error_pesudo_info = np.zeros([20, 21] , dtype=int)
# correct_pesudo_info = np.zeros([20, 20])
def parse_error_pesudo(det_bboxes_list, gts_bboxes_list):
    print(len(det_bboxes_list), len(gts_bboxes_list))
    for i in range(len(det_bboxes_list)):
        # print("det bboxes:", det_bboxes_list[i])
        # print("gts bboxes:", gts_bboxes_list[i])
        det_bboxes = torch.from_numpy(det_bboxes_list[i]['bboxes'])
        det_labels = torch.from_numpy(det_bboxes_list[i]['labels'])
        gts_bboxes = torch.from_numpy(gts_bboxes_list[i]['bboxes'])
        gts_labels = torch.from_numpy(gts_bboxes_list[i]['labels'])
        # overlaps_mask = cal_bboxes_multi_overlaps(det_bboxes, gts_bboxes, 0.75)



        overlaps, inds = cal_bboxes_overlaps(det_bboxes, gts_bboxes)
        thr_flag = overlaps > 0.5
        for i, over_flag in enumerate(thr_flag):
            if over_flag == False:
                error_pesudo_info[det_labels[i], 20] = error_pesudo_info[det_labels[i], 20] + 1

        inds = inds[thr_flag]
        if len(inds) == 0:
            continue

        flags = det_labels[thr_flag] == gts_labels[inds]
        for i, flag in enumerate(flags):
            if flag == False:
                error_pesudo_info[det_labels[thr_flag][i], gts_labels[inds][i]] = error_pesudo_info[det_labels[thr_flag][i], gts_labels[inds][i]] + 1

# pesudo_count_info = np.zeros(20)
# def parse_count_pesudo(det_bboxes_list, gts_bboxes_list):


#     data = pickle.load(f)
total_rank = 2
total_iter = range(49, 30050, 50)

data_dir ='/home/liu/ytx/SS-OD/outputs/BurnInTS/rcnn_voc_weight_0.4/pesudo_infos/'
outputs_dir = '/home/liu/ytx/SS-OD/outputs/pesudo_infos/'
pesudo_summary_list_path = outputs_dir + 'neg_region_recall.pkl'
error_pesudo_info_path = outputs_dir + 'error_info_50_neg_region_recall.pkl'

for iter_idx in total_iter:
    # iter_data_list = []
    det_bboxes_list = []
    gts_bboxes_list = []
    for rank_idx in range(total_rank):
        file_name = data_dir + 'iter_%d_rank_%d.pkl' % (iter_idx, rank_idx)
        file_data = read_pesudo_list(file_name)
        det_bboxes_list.extend(file_data['det_bboxes'])
        gts_bboxes_list.extend(file_data['gts_bboxes'])
    cal_summary(det_bboxes_list, gts_bboxes_list)
    parse_error_pesudo(det_bboxes_list, gts_bboxes_list)
    # exit()
    det_bboxes_list = []
    gts_bboxes_list = []
print("pesudo list:", pesudo_summary_list)
print(len(pesudo_summary_list))
print("error pesudo info:", error_pesudo_info)
# write_pesudo_list(pesudo_summary_list, pesudo_summary_list_path)
# write_pesudo_list(error_pesudo_info, error_pesudo_info_path)
print('finish')


