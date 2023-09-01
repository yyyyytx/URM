import torch
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.samplers import RandomSampler

@BBOX_SAMPLERS.register_module()
class PositiveOnlySampler(RandomSampler):

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Same to random sampler"""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        # print("positive:", pos_inds)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        return torch.tensor([], dtype=torch.int64)
        # neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        # if neg_inds.numel() != 0:
        #     neg_inds = neg_inds.squeeze(1)
        # print(assign_result)
        # if len(neg_inds) <= num_expected:
        #     return neg_inds
        # else:
        #     return self.random_choice(neg_inds, num_expected)