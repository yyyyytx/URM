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
total_iter = range(99, 30050, 100)



data_dir ='/home/liu/ytx/SS-OD/outputs/NegativeLoss/rcnn_voc_neg_unreliable_0.3_thr_0.0005_unreliable_con/proposal_regions_infos/'
outputs_dir = '/home/liu/ytx/SS-OD/outputs/roi_infos/'
pesudo_summary_list_path = outputs_dir + 'neg_unreliable_0.3_thr_0.0005_unreliable_con.pkl'
error_pesudo_info_path = outputs_dir + 'error_info_75_neg_unreliable_0.3_thr_0.0005_unreliable_con.pkl'

pos_num_list = []
pos_neg_num_list = []
neg_num_list = []
neg_pos_num_list = []

total_pos_region_num = 0
total_pos_neg_region_num = 0
total_neg_region_num = 0
total_neg_pos_region_num = 0
total_gts_region_num = 0

for iter_idx in total_iter:
    # iter_data_list = []
    pos_region_list = []
    pos_neg_list = []
    neg_region_list = []
    gts_bboxes_list = []
    pos_weight_list = []
    neg_weight_list = []

    pos_region_num = 0
    pos_neg_region_num = 0
    neg_region_num = 0
    neg_pos_region_num = 0
    gts_region_num = 0

    for rank_idx in range(total_rank):
        file_name = data_dir + 'iter_%d_rank_%d.pkl' % (iter_idx, rank_idx)
        file_data = read_pesudo_list(file_name)
        # print(file_data)
        pos_region_list.extend(file_data['pos_regions'])
        neg_region_list.extend(file_data['neg_regions'])
        gts_bboxes_list.extend(file_data['gts_regions'])



    for pos_bboxes, neg_bboxes, gts_bboxes in zip(pos_region_list, neg_region_list, gts_bboxes_list):
        # print(pos_bboxes, gts_bboxes)
        pos_overlaps, neg_overlaps = cal_unsup_sampling_overlaps(pos_bboxes[:, :],
                                                                 neg_bboxes[:, :],
                                                                 gts_bboxes)

        pos_region_num += torch.sum(pos_overlaps >= 0.5)
        pos_neg_region_num += torch.sum(pos_overlaps < 0.5)
        neg_region_num += torch.sum(neg_overlaps < 0.5)
        neg_pos_region_num += torch.sum(neg_overlaps >= 0.5)

    # print(true_pos_weights)
    print(pos_region_num, pos_neg_region_num, neg_region_num, neg_pos_region_num)
    pos_num_list.append(pos_region_num)
    pos_neg_num_list.append(pos_neg_region_num)
    neg_num_list.append(neg_region_num)
    neg_pos_num_list.append(neg_pos_region_num)

    total_pos_region_num += pos_region_num
    total_pos_neg_region_num += pos_neg_region_num
    total_neg_region_num += neg_region_num
    total_neg_pos_region_num += neg_pos_region_num

write_pesudo_list({'pos_num':pos_num_list,
                   'pos_neg_num':pos_neg_num_list,
                   'neg_num':neg_num_list,
                   'neg_pos_num':neg_pos_num_list},
                  pesudo_summary_list_path)
    # det_bboxes_list = []
    # gts_bboxes_list = []
# print("pesudo list:", pesudo_summary_list)
# write_pesudo_list(pesudo_summary_list, pesudo_summary_list_path)
print('total:', total_pos_region_num, total_pos_neg_region_num, total_neg_region_num, total_neg_pos_region_num)
print('finish')