import os
import pickle
import numpy as np

CLASSES = np.array(['aero', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'table', 'dog', 'horse',
               'motor', 'person', 'plant', 'sheep', 'sofa', 'train',
               'tv', 'BG'])

import matplotlib.pyplot as plt
eps = np.finfo(np.float32).eps
iou_thrs = [0.5, 0.75]
def read_pesudo_list(file):
    with open(file,'rb') as f:
        data =  pickle.load(f)
    return data

def format_latex_output(error_info):
    format_str=''
    for cls_id in range(20):
        # print("cls_id:", error_info[cls_id])
        format_str += '%s ' % CLASSES[cls_id]
        # print(error_info[cls_id])
        for count in error_info[cls_id]:
            format_str +='& %d ' % count
        format_str += '\\\\  \n'
        format_str += '\\hline \n'

    return format_str

outputs_dir = '/home/liu/ytx/SS-OD/outputs/pesudo_infos/'
burnin_path = outputs_dir + 'error_info_75_burnin_0.4.pkl'
# region_path = outputs_dir + 'region_error_info.pkl'
# region_reg_path = outputs_dir + 'error_info_75_region_negloss.pkl'
neg_path = outputs_dir + 'error_info_75_neg_unreliable_0.3_thr_0.0005.pkl'
negloss_region_path = outputs_dir + 'error_info_75_negloss_region.pkl'
# up_neg_path = outputs_dir + 'error_info_75_negloss_supneg.pkl'
sup_neg_region_path = outputs_dir + 'error_info_75_negloss_supneg_region.pkl'


# burnin_info = read_pesudo_list(burnin_path)
neg_info = read_pesudo_list(neg_path)
# negloss_region_info = read_pesudo_list(negloss_region_path)
# sup_neg_region_info = read_pesudo_list(sup_neg_region_path)

# print(neg_info[:, 20].tolist())
# print(negloss_region_info[:, 20].tolist())
# print(sup_neg_region_info[:, 20].tolist())
# print(burnin_info)
# burnin_str = format_latex_output(burnin_info)
# print('BurnIn')
# print(burnin_str)
neg_str = format_latex_output(neg_info)
print('Neg')
print(neg_str)
# negloss_region_str = format_latex_output(negloss_region_info)
# print('negloss_region')
# print(negloss_region_str)
#
# supneg_region_str = format_latex_output(sup_neg_region_info)
# print('Sup neg region')
# print(supneg_region_str)