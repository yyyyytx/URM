import torch
from mmdet.models import DETECTORS, FasterRCNN
from .BurnInTS import BurnInTSModel
from mmssod.utils.structure_utils import dict_split
from torch import nn
from mmssod.utils.gather import gather_same_shape_tensors, gather_diff_shape_tensors
# import torch.distributed as dist
from mmssod.models.utils.tensor_utils import norm_tensor
from torch.profiler import profile, record_function, ProfilerActivity
from ...utils.structure_utils import weighted_loss
from mmdet.core import build_sampler, build_assigner


@DETECTORS.register_module()
class PretrainFasterRCNN(FasterRCNN):
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # roi_losses = self.roi_head.kl_forward_train(x, img_metas, proposal_list,
        #                                          gt_bboxes, gt_labels,
        #                                          gt_bboxes_ignore, gt_masks,
        #                                          **kwargs)
        roi_losses = self.roi_head.iou_forward_train(x, img_metas, proposal_list,
                                                    gt_bboxes, gt_labels,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)
        # roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
        #                                             gt_bboxes, gt_labels,
        #                                             gt_bboxes_ignore, gt_masks,
        #                                             **kwargs)
        losses.update(roi_losses)

        return losses
