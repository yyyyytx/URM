import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
eps = np.finfo(np.float32).eps
iou_thrs = [0.5, 0.75]
def read_pesudo_list(file):
    with open(file,'rb') as f:
        data =  pickle.load(f)
    return data

def cal_recall_presious(pesudo_summary_list, iou_inx=0):
    recall = []
    precision = []
    fp = []
    tp = []
    gts = []
    for item in pesudo_summary_list:
        fp.append(item[iou_inx]['fp'])
        tp.append(item[iou_inx]['tp'])
        gts.append(item[iou_inx]['gts'])
        recall.append(item[iou_inx]['tp'] / np.maximum(item[iou_inx]['gts'], eps))
        precision.append(item[iou_inx]["tp"] / np.maximum((item[iou_inx]["tp"] + item[iou_inx]["fp"]), eps))
    return np.array(recall), np.array(precision), np.array(tp), np.array(fp), np.array(gts)


outputs_dir = '/home/liu/ytx/SS-OD/outputs/pesudo_infos/'
# filter_neg_path = outputs_dir + 'filter_neg.pkl'
# filter_pos_path = outputs_dir + 'filter_pos.pkl'
# region_est_path = outputs_dir + 'region_estimation.pkl'
# both_weight_path = outputs_dir + 'faster_rcnn_voc_both_weights.pkl'
# both_weight_ms = outputs_dir + 'faster_rcnn_voc_ms.pkl'
# unreliable_pos = outputs_dir + 'faster_rcnn_voc_unreliable_pos.pkl'
# unreliable_neg = outputs_dir + 'faster_rcnn_voc_unreliable_neg.pkl'

weight_0 = outputs_dir + 'burnin_weight_0.0.pkl'
weight_2 = outputs_dir + 'burnin_weight_0.2.pkl'
weight_4 = outputs_dir + 'burnin_0.4.pkl'
weight_6 = outputs_dir + 'burnin_weight_0.6.pkl'
weight_8 = outputs_dir + 'burnin_weight_0.8.pkl'
weight_10 = outputs_dir + 'burnin_weight_1.0.pkl'

# weight_0_recall, weight_0_precision, weight_0_tp, weight_0_fp, weight_0_gts = cal_recall_presious(read_pesudo_list(weight_0))
# weight_2_recall, weight_2_precision, weight_2_tp, weight_2_fp, weight_2_gts = cal_recall_presious(read_pesudo_list(weight_2))
weight_4_recall, weight_4_precision, weight_4_tp, weight_4_fp, weight_4_gts = cal_recall_presious(read_pesudo_list(weight_4))
# weight_6_recall, weight_6_precision, weight_6_tp, weight_6_fp, weight_6_gts = cal_recall_presious(read_pesudo_list(weight_6))
# weight_8_recall, weight_8_precision, weight_8_tp, weight_8_fp, weight_8_gts = cal_recall_presious(read_pesudo_list(weight_8))
# weight_10_recall, weight_10_precision, weight_10_tp, weight_10_fp, weight_10_gts = cal_recall_presious(read_pesudo_list(weight_10))
total_4 = weight_4_fp + weight_4_tp
colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']

# num = np.arange(0, len(weight_0_recall))
# plt.grid(linestyle='--')
# plt.plot(num, weight_0_recall, c=colors[0], label='0.0', linewidth=1)
# plt.plot(num, weight_2_recall, c=colors[1], label='0.2', linewidth=1)
# plt.plot(num, weight_4_recall, c=colors[2], label='0.4', linewidth=1)
# plt.plot(num, weight_6_recall, c=colors[3], label='0.6', linewidth=1)
# plt.plot(num, weight_8_recall, c=colors[4], label='0.8', linewidth=1)
# plt.plot(num, weight_10_recall, c=colors[5], label='1.0', linewidth=1)

# plt.plot(num, precions_keep_ratio, c='orange', label='keep ratio', linewidth=1)
# plt.ylabel('weights recalls')
# plt.legend(loc=4)
# plt.show()
#
# num = np.arange(0, len(weight_0_recall))
# plt.grid(linestyle='--')
# plt.plot(num, weight_0_precision, c=colors[0], label='0.0', linewidth=1)
# plt.plot(num, weight_2_precision, c=colors[1], label='0.2', linewidth=1)
# plt.plot(num, weight_4_precision, c=colors[2], label='0.4', linewidth=1)
# plt.plot(num, weight_6_precision, c=colors[3], label='0.6', linewidth=1)
# plt.plot(num, weight_8_precision, c=colors[4], label='0.8', linewidth=1)
# plt.plot(num, weight_10_precision, c=colors[5], label='1.0', linewidth=1)
#
# # plt.plot(num, precions_keep_ratio, c='orange', label='keep ratio', linewidth=1)
# plt.ylabel('weights precision')
# plt.legend(loc=4)
# plt.show()



# no_nms_path = outputs_dir + 'burnin_no_nms.pkl'
# filter_neg_path = outputs_dir + 'burnin_filter_neg.pkl'
# filter_pos_path = outputs_dir + 'burnin_filter_pos.pkl'
# region_path = outputs_dir + 'region.pkl'
# negloss_path = outputs_dir + 'negloss.pkl'
# negloss_region_path = outputs_dir + 'negloss_region.pkl'
# negloss_supneg_region_path = outputs_dir + 'negloss_supneg_region.pkl'
negloss_unreliable_05 = outputs_dir + 'neg_unreliable_0.5_thr_0.0005.pkl'
negloss_unreliable_05_unreliable_con = outputs_dir + 'neg_unreliable_0.5_thr_0.0005_unreliable_con.pkl'
negloss_unreliable_05_unreliable_con_without_unrelia = outputs_dir + 'neg_unreliable_0.5_thr_0.0005_unreliable_con_without_unrelia.pkl'

negloss_unreliable_03 = outputs_dir + 'neg_unreliable_0.3_thr_0.0005_unreliable_con_without_unrelia.pkl'
negloss_region_recall = outputs_dir + 'neg_region_recall.pkl'

# no_nms_recall, no_nms_precision, no_nms_tp, no_nms_fp, no_nms_gts = cal_recall_presious(read_pesudo_list(no_nms_path))
# filter_neg_recall, filter_neg_precision, filter_neg_tp, filter_neg_fp, filter_neg_gts = cal_recall_presious(read_pesudo_list(filter_neg_path))
# filter_pos_recall, filter_pos_precision, filter_pos_tp, filter_pos_fp, filter_pos_gts = cal_recall_presious(read_pesudo_list(filter_pos_path))
# region_recall, region_precision, region_tp, region_fp, region_gts = cal_recall_presious(read_pesudo_list(region_path))
# total_region = region_tp + region_fp
# negloss_recall, negloss_precision, negloss_tp, negloss_fp, _ = cal_recall_presious(read_pesudo_list(negloss_path))
# negloss_region_recall, negloss_region_precision, negloss_region_tp, negloss_region_fp, _ = cal_recall_presious(read_pesudo_list(negloss_region_path))
# negloss_supneg_region_recall, negloss_supneg_region_precision, negloss_supneg_region_tp, negloss_supneg_region_fp, _ = cal_recall_presious(read_pesudo_list(negloss_supneg_region_path))
negloss_unreliable_05_recall, negloss_unreliable_05_precision, negloss_unreliable_05_tp, negloss_unreliable_05_fp, _ = cal_recall_presious(read_pesudo_list(negloss_unreliable_05))
negloss_unreliable_05_unreliable_con_recall, negloss_unreliable_05_unreliable_con_precision, negloss_unreliable_05_unreliable_con_tp, negloss_unreliable_05_unreliable_con_fp, _ = cal_recall_presious(read_pesudo_list(negloss_unreliable_05_unreliable_con))
negloss_unreliable_05_unreliable_con_without_unrelia_recall, negloss_unreliable_05_unreliable_con_without_unrelia_precision, negloss_unreliable_05_unreliable_con_without_unrelia_tp, negloss_unreliable_05_unreliable_con_without_unrelia_fp, _ = cal_recall_presious(read_pesudo_list(negloss_unreliable_05_unreliable_con_without_unrelia))


negloss_unreliable_03_recall, negloss_unreliable_03_precision, negloss_unreliable_03_tp, negloss_unreliable_03_fp, _ = cal_recall_presious(read_pesudo_list(negloss_unreliable_03))

negloss_region_recall_recall, negloss_region_recall_precision, negloss_region_recall_tp, negloss_region_recall_fp, _ = cal_recall_presious(read_pesudo_list(negloss_region_recall))

print(len(negloss_unreliable_03_tp), len(negloss_region_recall_tp))

num = np.arange(0, len(weight_4_recall))
plt.grid(linestyle='--')
plt.plot(num, weight_4_recall, c=colors[0], label='0.4', linewidth=1)
# plt.plot(num, negloss_recall, c=colors[1], label='negloss', linewidth=1)
# plt.plot(num, negloss_region_recall, c=colors[2], label='negloss_region', linewidth=1)
plt.plot(num, negloss_unreliable_05_recall, c=colors[2], label='unreliable_05', linewidth=1)
plt.plot(num, negloss_unreliable_05_unreliable_con_recall, c=colors[1], label='unreliable_05_unreliable_con', linewidth=1)
plt.plot(num, negloss_unreliable_05_unreliable_con_without_unrelia_recall, c=colors[3], label='unreliable_05_unreliable_con_without_unrelia', linewidth=1)

# plt.plot(num, negloss_supneg_region_recall, c=colors[3], label='negloss_supneg_region', linewidth=1)

# plt.plot(num, no_nms_recall, c=colors[4], label='nms', linewidth=1)
# plt.plot(num, filter_neg_recall, c=colors[5], label='filter neg', linewidth=1)
# plt.plot(num, filter_pos_recall, c=colors[6], label='filter pos', linewidth=1)
# plt.plot(num, region_recall, c=colors[7], label='region', linewidth=1)
# plt.plot(num, negloss_recall, c=colors[8], label='negloss', linewidth=1)
# plt.plot(num, precions_keep_ratio, c='orange', label='keep ratio', linewidth=1)
plt.ylabel('recalls')
plt.legend(loc=4)
plt.show()


num = np.arange(0, len(weight_4_precision))
plt.grid(linestyle='--')
plt.plot(num, weight_4_precision, c=colors[0], label='0.4', linewidth=1)
# plt.plot(num, negloss_precision, c=colors[1], label='negloss', linewidth=1)
plt.plot(num, negloss_unreliable_05_precision, c=colors[2], label='unreliable_05', linewidth=1)
plt.plot(num, negloss_unreliable_05_unreliable_con_precision, c=colors[1], label='unreliable_05_unreliable_con', linewidth=1)
plt.plot(num, negloss_unreliable_05_unreliable_con_without_unrelia_precision, c=colors[3], label='unreliable_05_unreliable_con_without_unrelia', linewidth=1)

# plt.plot(num, negloss_region_precision, c=colors[2], label='negloss_region', linewidth=1)
# plt.plot(num, negloss_supneg_region_precision, c=colors[3], label='negloss_supneg_region', linewidth=1)

# plt.plot(num, no_nms_recall, c=colors[4], label='nms', linewidth=1)
# plt.plot(num, filter_neg_recall, c=colors[5], label='filter neg', linewidth=1)
# plt.plot(num, filter_pos_recall, c=colors[6], label='filter pos', linewidth=1)
# plt.plot(num, region_recall, c=colors[7], label='region', linewidth=1)
# plt.plot(num, negloss_recall, c=colors[8], label='negloss', linewidth=1)
# plt.plot(num, precions_keep_ratio, c='orange', label='keep ratio', linewidth=1)

# plt.plot(num, precions_keep_ratio, c='orange', label='keep ratio', linewidth=1)
plt.ylabel('precision')
plt.legend(loc=4)
plt.show()

# print('gts:')
# print(weight_6_gts.tolist())
# print(no_nms_gts.tolist())
# print(filter_neg_gts.tolist())
# print(filter_pos_gts.tolist())

num = np.arange(0, len(weight_4_tp))
plt.grid(linestyle='--')
plt.plot(num, weight_4_tp, c=colors[0], label='0.4', linewidth=1)
# plt.plot(num, negloss_tp, c=colors[1], label='negloss', linewidth=1)
# plt.plot(num, negloss_region_tp, c=colors[2], label='negloss_region', linewidth=1)
# plt.plot(num, negloss_supneg_region_tp, c=colors[3], label='negloss_supneg_region', linewidth=1)
plt.plot(num, negloss_unreliable_05_tp, c=colors[2], label='unreliable_05', linewidth=1)
plt.plot(num, negloss_unreliable_05_unreliable_con_tp, c=colors[1], label='unreliable_05_unreliable_con', linewidth=1)
plt.plot(num, negloss_unreliable_05_unreliable_con_without_unrelia_tp, c=colors[3], label='unreliable_05_unreliable_con_without_unrelia', linewidth=1)

plt.plot(num, negloss_region_recall_tp[1::2], c=colors[4], label='negloss_region_recall', linewidth=1)

# plt.plot(num, precions_keep_ratio, c='orange', label='keep ratio', linewidth=1)
plt.ylabel('tp')
plt.legend(loc=4)
plt.show()


num = np.arange(0, len(weight_4_fp))
plt.grid(linestyle='--')
plt.plot(num, weight_4_fp, c=colors[0], label='0.4', linewidth=1)
# plt.plot(num, negloss_fp, c=colors[1], label='negloss', linewidth=1)
# plt.plot(num, negloss_region_fp, c=colors[2], label='negloss_region', linewidth=1)
# plt.plot(num, negloss_supneg_region_fp, c=colors[3], label='negloss_supneg_region', linewidth=1)
plt.plot(num, negloss_unreliable_05_fp, c=colors[2], label='unreliable_05', linewidth=1)
plt.plot(num, negloss_unreliable_05_unreliable_con_fp, c=colors[1], label='unreliable_05_unreliable_con', linewidth=1)
plt.plot(num, negloss_unreliable_05_unreliable_con_without_unrelia_fp, c=colors[3], label='unreliable_05_unreliable_con_without_unrelia', linewidth=1)

plt.plot(num, negloss_region_recall_fp, c=colors[4], label='negloss_region_recall', linewidth=1)

# plt.plot(num, precions_keep_ratio, c='orange', label='keep ratio', linewidth=1)
plt.ylabel('fp')
plt.legend(loc=4)
plt.show()

print('tp:')
print(weight_4_tp.tolist())
# print(region_tp.tolist())
print(negloss_region_recall_tp.tolist())
# print(no_nms_tp.tolist())
# print(filter_neg_tp.tolist())
# print(filter_pos_tp.tolist())
#
# print('fp:')
# print(weight_4_fp.tolist())
# print(region_fp.tolist())
#
# print('total:')
# print(total_4.tolist())
# print(total_region.tolist())
# print(total_4.sum(), total_region.sum())

# print(no_nms_fp.tolist())
# print(filter_neg_fp.tolist())
# print(filter_pos_fp.tolist())


# filter_neg_recall, filter_neg_precision, filter_neg_tp, filter_neg_fp, filter_neg_gts = cal_recall_presious(read_pesudo_list(filter_neg_path))
# filter_pos_recall, filter_pos_precision, filter_pos_tp, filter_pos_fp, filter_pos_gts = cal_recall_presious(read_pesudo_list(filter_pos_path))

# region_est_recall, region_est_precision, region_est_tp, region_est_fp, _ = cal_recall_presious(read_pesudo_list(both_weight_path))
# region_est_ms_recall, region_est_ms_precision, region_est_ms_tp, region_est_ms_fp, _ = cal_recall_presious(read_pesudo_list(both_weight_ms))
# unreliable_pos_recall, unreliable_pos_precision, unreliable_pos_tp, unreliable_pos_fp, _ = cal_recall_presious(read_pesudo_list(unreliable_pos))
# unreliable_neg_recall, unreliable_neg_precision, unreliable_neg_tp, unreliable_neg_fp, _ = cal_recall_presious(read_pesudo_list(unreliable_neg))



