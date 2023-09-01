import torch

def bbox2mask(bbox_list):

    mask_list = []
    for (pos_num, neg_num) in bbox_list:
        pos_list = torch.ones(pos_num)
        neg_list = torch.zeros(neg_num)
        mask_list.append(pos_list)
        mask_list.append(neg_list)
    mask_list = torch.cat(mask_list, 0)
    return mask_list