import torch
from mmdet.models import DETECTORS
from .BurnInTS import BurnInTSModel
from ...utils.structure_utils import weighted_loss
from mmssod.core.bbox import PositiveOnlySampler
from mmdet.core import build_sampler, build_assigner

@DETECTORS.register_module()
class PosOnly(BurnInTSModel):
    def __init__(self, teacher: dict, student: dict, train_cfg=None, test_cfg=None, n_cls=20):
        super().__init__(teacher, student, train_cfg, test_cfg)
        # print("train_cfg:",train_cfg)
        # print(train_cfg.unsup_sampler)
        self.rpn_bbox_unsup_sampler = build_sampler(train_cfg.unsup_sampler)
        self.rcnn_bbox_unsup_sampler = build_sampler(train_cfg.unsup_sampler)
        self.rpn_bbox_unsup_assigner = build_assigner(train_cfg.assigner)
        self.rcnn_bbox_unsup_assigner = build_assigner(train_cfg.assigner)

        self.rpn_bbox_sup_sampler = build_sampler(self.student.train_cfg.rpn.sampler)
        self.rcnn_bbox_sup_sampler = build_sampler(self.student.train_cfg.rcnn.sampler)
        self.rpn_bbox_sup_assigner = build_assigner(self.student.train_cfg.rpn.assigner)
        self.rcnn_bbox_sup_assigner = build_assigner(self.student.train_cfg.rcnn.assigner)


    def _compute_unsup_loss(self, sup_data, unsup_data):
        strong_unsup = self._gen_pseudo_labels(unsup_data)

        losses = dict()
        self._switch_sup_train()
        sup_loss = self.student.forward_train(**sup_data)
        sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
        losses.update(sup_loss)

        self._switch_unsup_train()
        # unsup_loss = self.student.forward_train(**strong_unsup)
        unsup_loss = self._compute_student_unsup_losses(strong_unsup)
        unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}

        losses.update(unsup_loss)
        # losses = weighted_loss(losses, self.unsup_loss_weight)
        return losses

    def _compute_student_unsup_losses(self, strong_unsup, proposals=None):
        feat = self.student.extract_feat(strong_unsup["img"])
        losses = dict()
        # student RPN forward and loss
        if self.student.with_rpn:
            proposal_cfg = self.student.train_cfg.get('rpn_proposal',
                                                      self.student.test_cfg.rpn)
            rpn_losses, proposal_list = self.student.rpn_head.forward_train(
                feat,
                strong_unsup["img_metas"],
                strong_unsup["gt_bboxes"],
                gt_labels=None,
                gt_bboxes_ignore=None,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.student.roi_head.forward_train(feat,
                                                         strong_unsup["img_metas"],
                                                         proposal_list,
                                                         strong_unsup["gt_bboxes"],
                                                         strong_unsup["gt_labels"])
        losses.update(roi_losses)


        # print(feat)
        return self._check_losses_item(losses, feat[0].device)

    def _check_losses_item(self, losses, device):
        items_list = ["loss_rpn_cls", "loss_rpn_bbox", "loss_cls", "loss_bbox", "acc"]
        for item in items_list:
            if item not in losses.keys():
                losses[item] = torch.tensor(0.).to(device)
        return losses


    def _switch_sup_train(self):
        self.student.rpn_head.sampler = self.rpn_bbox_sup_sampler
        self.student.roi_head.bbox_sampler = self.rcnn_bbox_sup_sampler
        self.student.rpn_head.assigner = self.rpn_bbox_sup_assigner
        self.student.roi_head.bbox_assigner = self.rcnn_bbox_sup_assigner

    def _switch_unsup_train(self):
        self.student.rpn_head.sampler = self.rpn_bbox_unsup_sampler
        self.student.roi_head.bbox_sampler = self.rcnn_bbox_unsup_sampler
        self.student.rpn_head.assigner = self.rpn_bbox_unsup_assigner
        self.student.roi_head.bbox_assigner = self.rcnn_bbox_unsup_assigner
