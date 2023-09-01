import os
import pickle
from mmssod.models.utils.eval_utils import cal_recall_precisions, cal_unsup_sampling_overlaps, cal_bboxes_overlaps
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



data_dir ='/home/liu/ytx/SS-OD/outputs/NegativeLoss/rcnn_voc_neg_unreliable_0.3_thr_0.0005_unreliable_con/neg_unreliable_infos/'
outputs_dir = '/home/liu/ytx/SS-OD/outputs/roi_infos/'
# pesudo_summary_list_path = outputs_dir + 'negloss_unreliable_0.3_thr_0.0005_unreliable_con_without_unrelia.pkl'
# weight_roi_path = outputs_dir + 'negloss_unreliable_0.5_thr_0.0005_roi_weight.pkl'

def split_tensors(data, split):
    l = []
    cur_ind = 0
    for i in range(len(split)):
        l.append(data[cur_ind:cur_ind+split[i]])
        cur_ind += split[i]
    return l

total_rank = 2
total_iter = range(49, 30050, 50)

# 实际是背景却负优化与背景的相似度
bg_error_count = 0
# 实际是背景却优化与前景的相似度
bg_fg_count = 0


fg_error_count = 0
fg_bg_count = 0

total_count = 0
for iter_idx in total_iter:
    # iter_data_list = []
    unreliable_bboxes_list = []
    gts_bboxes_list = []
    unraliable_mask_list = []

    for rank_idx in range(total_rank):
        file_name = data_dir + 'iter_%d_rank_%d.pkl' % (iter_idx, rank_idx)
        file_data = read_pesudo_list(file_name)
        for unraliable_bboxes, unreliable_mask in zip(file_data['unraliable_bboxes'],file_data['unraliable_mask']):
            unreliable_bboxes_list.extend(unraliable_bboxes)
            # unraliable_mask_list.extend(file_data['unraliable_mask'][])
            # print(len(unraliable_bboxes))

            unraliable_mask_list.extend(split_tensors(unreliable_mask, [len(unraliable_bboxes[0]), len(unraliable_bboxes[1])]))
        gts_bboxes_list.extend(file_data['gts_bboxes'])
        # print(file_data['gts_bboxes'])
        # gts_labels_list.extend(file_data['gts_labels'])
        # unreliable_bboxes_list.extend(split_tensors(file_data['unraliable_mask', []]))
        # unraliable_mask_list.extend(file_data['unraliable_mask'])

    # print(len(unreliable_bboxes_list))
    # print(gts_bboxes_list[0])
    print(len(unreliable_bboxes_list), len(gts_bboxes_list), len(unraliable_mask_list))

    for idx in range(len(unreliable_bboxes_list)):
        overlaps, inds = cal_bboxes_overlaps(torch.from_numpy(unreliable_bboxes_list[idx]), torch.from_numpy(gts_bboxes_list[idx]['bboxes']))
        unreliable_labels = torch.from_numpy(gts_bboxes_list[idx]['labels'])
        if len(inds) == 0:
            continue

        overlap_mask = overlaps > 0.5
        # print(unraliable_mask_list[idx])
        for i in range(len(overlap_mask)):
            if overlap_mask[i] == False:
                if unraliable_mask_list[idx][i][20] == True:
                    bg_error_count += 1
                if unraliable_mask_list[idx][i].sum() != 0:
                    bg_fg_count += 1

            else:
                if unraliable_mask_list[idx][i][unreliable_labels[inds[i]]] == True:
                    fg_error_count += 1
                if unraliable_mask_list[idx][i][20] == True:
                    fg_bg_count += 1

            total_count += 1
        print(bg_error_count, bg_fg_count ,fg_error_count, fg_bg_count,total_count)
        # print(unraliable_mask_list[idx])
        # print(unreliable_labels[inds])
        # print(overlaps)



    unreliable_bboxes_list = []
    gts_bboxes_list = []
    unraliable_mask_list = []

print('total:', fg_error_count, fg_bg_count,bg_error_count, bg_fg_count ,total_count)

