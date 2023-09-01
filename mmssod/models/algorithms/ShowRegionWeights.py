import torch
from mmdet.models import DETECTORS
from .RegionEstimation import RegionEstimation
from mmssod.utils.structure_utils import dict_split
from torch import nn

@DETECTORS.register_module()
class ShowRegionWeights(RegionEstimation):

    def _compute_student_rcnn_losses(self,
                                     x,
                                     img_metas,
                                     proposal_list,
                                     gt_bboxes,
                                     gt_labels,
                                     gt_bboxes_ignore=None,
                                     gt_masks=None,
                                     gt_bboxes_true=None,
                                     gt_labels_true=None,
                                     **kwargs):
        if self.student.roi_head.with_bbox or self.student.roi_head.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.student.roi_head.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                # print("assign_result:", assign_result)
                sampling_result = self.student.roi_head.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                # print("sample result:", sampling_result)
                if self.print_pesudo_summary:
                    self._add_unsup_sampling_bboxes(sampling_result.pos_bboxes.clone(),
                                                    sampling_result.neg_bboxes.clone(),
                                                    gt_bboxes_true[i].detach().cpu().numpy())

                sampling_results.append(sampling_result)

        losses = dict()
        if self.student.roi_head.with_bbox:
            bbox_results, rois, label_weights = self.student.roi_head._weighted_bbox_forward_train(x,
                                                                              sampling_results,
                                                                              gt_bboxes,
                                                                              gt_labels,
                                                                              img_metas,
                                                                              self.pos_queue,
                                                                              self.neg_queue,
                                                                              self.is_unreliable,
                                                                              self.unreliable_pos_queue,
                                                                              self.unreliable_neg_queue,
                                                                              self.n_cls,
                                                                              True)

            for i in range(len(proposal_list)):
                idx = rois[:, 0] == i
                print(idx)
            print("rois:", rois.shape, rois)
            print("labels weights:", label_weights.shape)
            losses.update(bbox_results['loss_bbox'])
            if self.is_unreliable is True:
                losses.update(unreliable_loss=bbox_results['unreliable_loss'])

        return losses