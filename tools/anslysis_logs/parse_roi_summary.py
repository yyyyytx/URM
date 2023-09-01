import os
import pickle
from mmssod.models.utils.eval_utils import cal_recall_precisions, cal_unsup_sampling_overlaps
import numpy as np
eps = np.finfo(np.float32).eps
import torch
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def read_pesudo_list(file):
    # with torch.loading_context(map_location='cpu'):
    with open(file,'rb') as f:
        data = CPU_Unpickler(f).load()
    return data

def write_pesudo_list(pesudo_list, file):
    with open(file,'wb') as f:
        pickle.dump(pesudo_list, f)


total_rank = 2
total_iter = range(49, 90000, 50)


data_dir ='/home/liu/ytx/SS-OD/outputs/NegativeLoss/90k_rcnn_voc_region_neg_without_weight/proposal_regions_infos/'
outputs_dir = '/home/liu/ytx/SS-OD/outputs/roi_infos/'
pesudo_summary_list_path = outputs_dir + '90k_region_neg.pkl'
weight_roi_path = outputs_dir + '90k_region_neg_without_weight.pkl'

pos_num_list = []
pos_neg_num_list = []
neg_num_list = []
neg_pos_num_list = []
true_pos_weights_list = []
false_pos_weights_list = []
true_neg_weights_list = []
false_neg_weights_list = []
for iter_idx in total_iter:
    # iter_data_list = []
    pos_region_list = []
    neg_region_list = []
    gts_bboxes_list = []
    pos_weight_list = []
    neg_weight_list = []
    pos_region_num = 0
    pos_neg_region_num = 0
    neg_region_num = 0
    neg_pos_region_num = 0
    gts_region_num = 0
    true_pos_weights = []
    false_pos_weights = []
    true_neg_weights = []
    false_neg_weights = []


    for rank_idx in range(total_rank):
        file_name = data_dir + 'iter_%d_rank_%d.pkl' % (iter_idx, rank_idx)
        file_data = read_pesudo_list(file_name)
        # print(file_data)
        pos_region_list.extend(file_data['pos_regions'])
        neg_region_list.extend(file_data['neg_regions'])
        gts_bboxes_list.extend(file_data['gts_regions'])
        pos_weight_list.extend(file_data['pos_weights'])
        neg_weight_list.extend(file_data['neg_weights'])


    for pos_bboxes, neg_bboxes, gts_bboxes, pos_weights, neg_weights in zip(pos_region_list, neg_region_list, gts_bboxes_list, pos_weight_list, neg_weight_list):
        # print(pos_bboxes, gts_bboxes)
        pos_overlaps, neg_overlaps = cal_unsup_sampling_overlaps(pos_bboxes[:, 1:],
                                                                 neg_bboxes[:, 1:],
                                                                 gts_bboxes)
        # print(pos_overlaps)
        pos_region_num += torch.sum(pos_overlaps >= 0.5)
        true_pos_weights.append(torch.mean(pos_weights[pos_overlaps >= 0.25]))
        false_pos_weights.append(torch.mean(pos_weights[pos_overlaps < 0.25]))
        pos_neg_region_num += torch.sum(pos_overlaps < 0.5)
        neg_region_num += torch.sum(neg_overlaps < 0.5)
        neg_pos_region_num += torch.sum(neg_overlaps >= 0.5)
        true_neg_weights.append(torch.mean(neg_weights[neg_overlaps < 0.75]))
        false_neg_weights.append(torch.mean(neg_weights[neg_overlaps >= 0.75]))

    # print(true_pos_weights)
    print(pos_region_num, pos_neg_region_num, neg_region_num, neg_pos_region_num)
    true_pos_weights = torch.tensor(true_pos_weights)
    false_pos_weights = torch.tensor(false_pos_weights)
    true_neg_weights = torch.tensor(true_neg_weights)
    false_neg_weights = torch.tensor(false_neg_weights)

    pos_num_list.append(pos_region_num)
    pos_neg_num_list.append(pos_neg_region_num)
    neg_num_list.append(neg_region_num)
    neg_pos_num_list.append(neg_pos_region_num)

    # print(torch.isnan(true_pos_weights))
    print("weights:", torch.mean(true_pos_weights[~torch.isnan(true_pos_weights)]),
          torch.mean(false_pos_weights[~torch.isnan(false_pos_weights)]),
          torch.mean(true_neg_weights[~torch.isnan(true_neg_weights)]),
          torch.mean(false_neg_weights[~torch.isnan(false_neg_weights)]))

    true_pos_weights_list.append(torch.mean(true_pos_weights[~torch.isnan(true_pos_weights)]))
    false_pos_weights_list.append(torch.mean(false_pos_weights[~torch.isnan(false_pos_weights)]))
    true_neg_weights_list.append(torch.mean(true_neg_weights[~torch.isnan(true_neg_weights)]))
    false_neg_weights_list.append(torch.mean(false_neg_weights[~torch.isnan(false_neg_weights)]))
    # cal_summary(det_bboxes_list, gts_bboxes_list)
    # det_bboxes_list = []
    # gts_bboxes_list = []
# print("pesudo list:", pesudo_summary_list)
# write_pesudo_list(pesudo_summary_list, pesudo_summary_list_path)

# write_pesudo_list({'pos_num':pos_num_list,
#                    'pos_neg_num':pos_neg_num_list,
#                    'neg_num':neg_num_list,
#                    'neg_pos_num':neg_pos_num_list},
#                   pesudo_summary_list_path)

write_pesudo_list({"true_pos_weights" : true_pos_weights_list,
                   "false_pos_weights" : false_pos_weights_list,
                   "true_neg_weights" : true_neg_weights_list,
                   "false_neg_weights" : false_neg_weights_list},
                  weight_roi_path)
print('finish')