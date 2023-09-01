import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
eps = np.finfo(np.float32).eps
def read_pesudo_list(file):
    with open(file,'rb') as f:
        data =  pickle.load(f)
    return data

def cal_summary(pesudo_summary_dict):
    pos_num_list = pesudo_summary_dict['pos_num']
    pos_neg_num_list = pesudo_summary_dict['pos_neg_num']
    neg_num_list = pesudo_summary_dict['neg_num']
    neg_pos_num_list = pesudo_summary_dict['neg_pos_num']

    return np.array(pos_num_list), np.array(pos_neg_num_list), np.array(neg_num_list), np.array(neg_pos_num_list)

def cal_roi_weight(weight_dict):
    print(weight_dict)
    true_pos_weights = weight_dict['true_pos_weights']
    false_pos_weights = weight_dict['false_pos_weights']
    true_neg_weights = weight_dict['true_neg_weights']
    false_neg_weights = weight_dict['false_neg_weights']
    return np.array(true_pos_weights), np.array(false_pos_weights), np.array(true_neg_weights), np.array(false_neg_weights)

outputs_dir = '/home/liu/ytx/SS-OD/outputs/roi_infos/'

weight_6 = outputs_dir + 'burnin.pkl'
negloss_region = outputs_dir + 'negloss_region.pkl'
negloss_supneg_region = outputs_dir + 'negloss_supneg_region.pkl'

negloss_region_weight_path = outputs_dir + 'negloss_region_roi_weight.pkl'
negloss_supneg_region_weight_path = outputs_dir + 'negloss_supneg_region_roi_weight.pkl'


region_neg_without_weight = outputs_dir + '90k_region_neg_without_weight.pkl'

# weight_6_true_pos, weight_6_false_pos, weight_6_true_neg, weight_6_false_neg = cal_summary(read_pesudo_list(weight_6))
# negloss_region_true_pos, negloss_region_false_pos, negloss_region_true_neg, negloss_region_false_neg = cal_summary(read_pesudo_list(negloss_region))
# negloss_supneg_region_true_pos, negloss_supneg_region_false_pos, negloss_supneg_region_true_neg, negloss_supneg_region_false_neg = cal_summary(read_pesudo_list(negloss_supneg_region))

# negloss_region_true_pos_weight, negloss_region_false_pos_weight, negloss_region_true_neg_weight, negloss_region_false_neg_weight = cal_roi_weight(read_pesudo_list(negloss_region_weight_path))
# negloss_supneg_region_true_pos_weight, negloss_supneg_region_false_pos_weight, negloss_supneg_region_true_neg_weight, negloss_supneg_region_false_neg_weight = cal_roi_weight(read_pesudo_list(negloss_supneg_region_weight_path))

region_neg_without_weight_true_pos_weight, region_neg_without_weight_false_pos_weight,region_neg_without_weight_true_neg_weight , region_neg_without_weight_false_neg_weight = cal_roi_weight(read_pesudo_list(region_neg_without_weight))
# print(region_neg_without_weight_true_pos_weight)
colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']

num = np.arange(0, len(region_neg_without_weight_true_pos_weight))
plt.grid(linestyle='--')
plt.plot(num, region_neg_without_weight_true_pos_weight, c=colors[0], label='true pos', linewidth=1)
plt.plot(num, region_neg_without_weight_false_pos_weight, c=colors[1], label='false pos', linewidth=1)

# plt.plot(num, region_neg_without_weight_true_neg_weight, c=colors[2], label='true neg', linewidth=1)
# plt.plot(num, region_neg_without_weight_false_neg_weight, c=colors[3], label='false neg', linewidth=1)


plt.legend(loc=4)
plt.show()



num = np.arange(0, len(region_neg_without_weight_true_pos_weight))
plt.grid(linestyle='--')
# plt.plot(num, region_neg_without_weight_true_pos_weight, c=colors[0], label='true pos', linewidth=1)
# plt.plot(num, region_neg_without_weight_false_pos_weight, c=colors[1], label='false pos', linewidth=1)

plt.plot(num, region_neg_without_weight_true_neg_weight, c=colors[2], label='true neg', linewidth=1)
plt.plot(num, region_neg_without_weight_false_neg_weight, c=colors[3], label='false neg', linewidth=1)


plt.legend(loc=4)
plt.show()

# num = np.arange(0, len(weight_6_true_pos))
# plt.grid(linestyle='--')
# plt.plot(num, weight_6_true_pos, c=colors[0], label='0.4', linewidth=1)
# plt.plot(num, negloss_supneg_region_true_pos, c=colors[1], label='negloss supneg region', linewidth=1)
# plt.plot(num, negloss_region_true_pos, c=colors[2], label='negloss region', linewidth=1)
#
# plt.ylabel('true_pos')
# plt.legend(loc=4)
# plt.show()
#
# num = np.arange(0, len(weight_6_false_pos))
# plt.grid(linestyle='--')
# plt.plot(num, weight_6_false_pos, c=colors[0], label='0.4', linewidth=1)
# plt.plot(num, negloss_supneg_region_false_pos, c=colors[1], label='negloss supneg region', linewidth=1)
# plt.plot(num, negloss_region_false_pos, c=colors[2], label='negloss region', linewidth=1)
#
#
# # plt.plot(num, precions_keep_ratio, c='orange', label='keep ratio', linewidth=1)
# plt.ylabel('false_pos')
# plt.legend(loc=4)
# plt.show()
#
# num = np.arange(0, len(negloss_region_true_pos_weight))
# plt.grid(linestyle='--')
# plt.plot(num, negloss_region_true_pos_weight, c=colors[0], label='true_pos', linewidth=1)
# plt.plot(num, negloss_region_false_pos_weight, c=colors[1], label='false_pos', linewidth=1)
# plt.plot(num, negloss_region_true_neg_weight, c=colors[2], label='true_neg', linewidth=1)
# plt.plot(num, negloss_region_false_neg_weight, c=colors[3], label='false_neg', linewidth=1)
#
#
# # plt.plot(num, precions_keep_ratio, c='orange', label='keep ratio', linewidth=1)
# plt.ylabel('negloss_region')
# plt.legend(loc=4)
# plt.show()
#
# num = np.arange(0, len(negloss_supneg_region_true_pos_weight))
# plt.grid(linestyle='--')
# plt.plot(num, negloss_supneg_region_true_pos_weight, c=colors[0], label='true_pos', linewidth=1)
# plt.plot(num, negloss_supneg_region_false_pos_weight, c=colors[1], label='false_pos', linewidth=1)
# plt.plot(num, negloss_supneg_region_true_neg_weight, c=colors[2], label='true_neg', linewidth=1)
# plt.plot(num, negloss_supneg_region_false_neg_weight, c=colors[3], label='false_neg', linewidth=1)
#
#
# # plt.plot(num, precions_keep_ratio, c='orange', label='keep ratio', linewidth=1)
# plt.ylabel('negloss_supneg_region')
# plt.legend(loc=4)
# plt.show()
#
# print('negloss_region')
# print(negloss_region_true_pos_weight.tolist())
# print(negloss_region_false_pos_weight.tolist())
# print(negloss_region_true_neg_weight.tolist())
# print(negloss_region_false_neg_weight.tolist())
#
# print('negloss_supneg_region')
# print(negloss_supneg_region_true_pos_weight.tolist())
# print(negloss_supneg_region_false_pos_weight.tolist())
# print(negloss_supneg_region_true_neg_weight.tolist())
# print(negloss_supneg_region_false_neg_weight.tolist())