from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead
import torch
from mmdet.core import (bbox2roi, roi2bbox)
from mmssod.core.bbox.transforms import bbox2mask
# import torch.nn.functional as F
from torch import nn
from mmdet.core import multiclass_nms
from mmdet.core.bbox.builder import build_assigner, build_sampler
from mmssod.models.utils.eval_utils import cal_bboxes_overlaps
from mmdet.core import multiclass_nms
from torch import cdist
from mmssod.models.utils.tensor_utils import norm_tensor, norm_tensor2
from torch import nn
from mmdet.models.losses import CrossEntropyLoss
from mmssod.models.utils.bbox_utils import Transform2D
import torch.nn.functional as F
from mmcv.runner.fp16_utils import force_fp32
from mmssod.models.utils.tensor_utils import norm_tensor

eps: float = 1e-7


@HEADS.register_module()
class RoiFeaturesHead(StandardRoIHead):
    def __init__(self, **kwargs):
        super(RoiFeaturesHead, self).__init__(**kwargs)
        self.pic_ind = 0

    # def _bbox_feat_forward(self, x, rois):
    #     """Box head forward function used in both training and testing."""
    #     # TODO: a more flexible way to decide which feature maps to use
    #     bbox_feats = self.bbox_roi_extractor(
    #         x[:self.bbox_roi_extractor.num_inputs], rois)
    #     if self.with_shared_head:
    #         bbox_feats = self.shared_head(bbox_feats)
    #     cls_score, bbox_pred, projector_feats, feats, iou_pred = self.bbox_head.forward_feats(bbox_feats)
    #
    #     bbox_results = dict(
    #         cls_score=cls_score, bbox_pred=bbox_pred)
    #     return bbox_results, projector_feats, feats, iou_pred

    def extract_bboxes_features(self,
                                x,
                                bboxes):
        # _, _, bbox_feats, iou_pred = self._bbox_feat_forward(x, bboxes)
        results = self._bbox_iou_forward(x, bboxes)
        return results['iou_pred']
    #
    # def extract_projector_features(self,
    #                                x,
    #                                proposals,
    #                                gt_bboxes,
    #                                train_cfg,
    #                                bg_score_thr=0.3):
    #     bg_roi_list = []
    #     for i in range(len(proposals)):
    #         # print('proposals:', proposals[i])
    #         rois = proposals[i][:, :4]
    #         roi_scores = proposals[i][:, 4]
    #         overlaps, _ = cal_bboxes_overlaps(rois, gt_bboxes[i], mode='iof')
    #         rois = rois[overlaps < train_cfg.region_bg_iou_thr]
    #         roi_scores = roi_scores[overlaps < train_cfg.region_bg_iou_thr].view(-1, 1).repeat(1, 2)
    #         bg_rois, _ = multiclass_nms(rois,
    #                                     roi_scores,
    #                                     bg_score_thr,
    #                                     train_cfg.region_bg_nms_cfg,
    #                                     max_num=train_cfg.region_bg_max_num)
    #         img_inds = bg_rois.new_full((bg_rois.size(0), 1), i)
    #         bg_rois = torch.cat([img_inds, bg_rois[:, :4]], dim=-1)
    #
    #         bg_roi_list.append(bg_rois)
    #     bg_rois = torch.cat(bg_roi_list, 0)
    #     _, bg_projector_feats, bg_feats, _ = self._bbox_feat_forward(x, bg_rois)
    #
    #     gt_roi_list = []
    #     for i in range(len(gt_bboxes)):
    #         gt_rois = gt_bboxes[i]
    #         img_inds = gt_rois.new_full((gt_rois.size(0), 1), i)
    #         gt_rois = torch.cat([img_inds, gt_rois[:, :4]], dim=-1)
    #
    #         gt_roi_list.append(gt_rois)
    #     gt_rois = torch.cat(gt_roi_list, 0)
    #     _, gt_projector_feats, gt_feats,_ = self._bbox_feat_forward(x, gt_rois)
    #     return bg_projector_feats, gt_projector_feats, bg_feats, gt_feats
    #
    #
    # def _sup_bbox_forward_train(self,
    #                             x,
    #                             sampling_results,
    #                             assign_results,
    #                             gt_bboxes,
    #                             gt_labels,
    #                             img_metas):
    #
    #
    #     rois = bbox2roi([res.bboxes for res in sampling_results])
    #     bbox_results, _, _, iou_pred = self._bbox_feat_forward(x, rois)
    #     # print(bbox_results["bbox_pred"].shape)
    #     # print(torch.sigmoid(iou_pred).shape)
    #     bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
    #                                               gt_labels, self.train_cfg)
    #     loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
    #                                     bbox_results['bbox_pred'], rois,
    #                                     *bbox_targets)
    #
    #
    #
    #
    #     bbox_results.update(loss_bbox=loss_bbox)
    #     return bbox_results
    #
    # def _weighted_bbox_forward_train(self,
    #                                  x,
    #                                  sampling_results,
    #                                  gt_bboxes,
    #                                  gt_labels,
    #                                  pos_queue,
    #                                  neg_queue,
    #                                  n_cls=20,
    #                                  return_weights=False):
    #     rois = bbox2roi([res.bboxes for res in sampling_results])
    #     rois_mask = bbox2mask([(len(res.pos_bboxes), len(res.neg_bboxes)) for res in sampling_results])
    #
    #     bbox_results, bboxes_weighted_feats, _, _ = self._bbox_feat_forward(x, rois)
    #     # print("bbox_results:", bbox_results['cls_score'].shape, bbox_results['bbox_pred'].shape, bboxes_weighted_feats.shape)
    #     # print(bbox_results['cls_score'])
    #
    #     bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
    #                                               gt_labels, self.train_cfg)
    #     # print(bboxes_weighted_feats.shape, rois.shape)
    #     # print("bbox_targets:", bbox_targets)
    #     # neg labels: voc 20
    #     bg_mask = bbox_targets[0] == n_cls
    #     fg_mask = bbox_targets[0] != n_cls
    #
    #     normalize_feat = nn.functional.normalize(bboxes_weighted_feats, dim=0)
    #
    #     norm_feats = norm_tensor(normalize_feat)
    #     # norm_neg_queue = norm_tensor2(neg_queue)
    #     norm_neg_queue = neg_queue
    #     bg_sim = torch.mm(norm_feats, norm_neg_queue)
    #
    #     pos_queue = pos_queue.permute(1, 0, 2).reshape(norm_feats.shape[1], -1)
    #     # norm_pos_queue = norm_tensor2(pos_queue)
    #     norm_pos_queue = pos_queue
    #     fg_sim = torch.mm(norm_feats, norm_pos_queue)
    #     bg_sim_values, bg_sim_inds = torch.max(bg_sim, dim=1)
    #     fg_sim_values, fg_sim_inds = torch.max(fg_sim, dim=1)
    #
    #     def min_max_norm(x):
    #         if len(x) == 0 or len(x) == 1:
    #             return x
    #         # print("x:", x)
    #         min = torch.min(x, dim=0).values.detach()
    #         max = torch.max(x, dim=0).values.detach()
    #         # print('max:', max, min)
    #         return ((x- min) + eps) / ((max - min) + eps)
    #
    #     bg_cls_weights = torch.mul(torch.clamp(bg_sim_values[bg_mask], min=0.), 1-fg_sim_values[bg_mask])
    #     fg_cls_weights = torch.mul(torch.clamp(fg_sim_values[fg_mask], min=0.), 1-bg_sim_values[fg_mask])
    #
    #     bg_cls_weights = min_max_norm(bg_cls_weights)
    #     fg_cls_weights = min_max_norm(fg_cls_weights)
    #
    #     cls_weights = torch.zeros(len(bbox_results["cls_score"])).to(bbox_results["cls_score"].device).half()
    #     cls_weights = torch.masked_scatter(cls_weights, bg_mask, bg_cls_weights).detach()
    #     cls_weights = torch.masked_scatter(cls_weights, fg_mask, fg_cls_weights).detach()
    #
    #     bboxes = roi2bbox(rois)
    #     total_overlaps = []
    #
    #     for i in range(len(gt_bboxes)):
    #         overlaps, _ = cal_bboxes_overlaps(bboxes[i], gt_bboxes[i])
    #         total_overlaps.append(overlaps)
    #         # print(overlaps)
    #     total_overlaps = torch.cat(total_overlaps)
    #     # print(total_overlaps)
    #
    #     def bbox_weight(ious):
    #         ious = (ious - 0.5) / 0.2
    #         return 1. / (1. + torch.exp(-15. * (2 * ious - 1)))
    #
    #     bbox_weights = torch.ones(len(total_overlaps)).to(total_overlaps.device)
    #     mask = (total_overlaps > 0.5)
    #     # 1. / (1. + torch.exp(-15. * (2 * ious - 1)))
    #     # re_scores = 1. / (1. + torch.exp(-15. * total_overlaps[mask]))
    #     re_scores = bbox_weight(total_overlaps[mask])
    #     bbox_weights = torch.masked_scatter(bbox_weights, mask, re_scores)
    #     bbox_weights = bbox_weights.reshape((-1, 1)).repeat(1, 4).detach()
    #     # bbox_weights = regression_reweight(total_overlaps).reshape((-1, 1)).repeat(1, 4).detach()
    #     # print(bbox_weights)
    #
    #
    #
    #     # [cal_bboxes_overlaps()]
    #     # overlaps, _ = cal_bboxes_overlaps(rois, gt_bboxes)
    #
    #
    #     # bbox_weights = cls_weights.reshape((-1, 1)).repeat(1, 4).detach()
    #     loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
    #                                     bbox_results['bbox_pred'],
    #                                     rois,
    #                                     bbox_targets[0],
    #                                     cls_weights,
    #                                     bbox_targets[2],
    #                                     bbox_weights)
    #
    #     bbox_results.update(loss_bbox=loss_bbox)
    #     if return_weights == False:
    #         return bbox_results
    #     else:
    #         return bbox_results, rois, rois_mask, cls_weights, bbox_targets[0]
    #
    # def _center_weighted_bbox_forward_train(self,
    #                                  x,
    #                                  sampling_results,
    #                                  gt_bboxes,
    #                                  gt_labels,
    #                                  center_feats,
    #                                  n_cls=20,
    #                                  return_weights=False,
    #                                         is_weight_norm=True):
    #     rois = bbox2roi([res.bboxes for res in sampling_results])
    #     rois_mask = bbox2mask([(len(res.pos_bboxes), len(res.neg_bboxes)) for res in sampling_results])
    #
    #     bbox_results, _, bboxes_weighted_feats, iou_pred = self._bbox_feat_forward(x, rois)
    #     # print("bbox_results:", bbox_results['cls_score'].shape, bbox_results['bbox_pred'].shape, bboxes_weighted_feats.shape)
    #     # print(bbox_results['cls_score'])
    #
    #     bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
    #                                               gt_labels, self.train_cfg)
    #     # print(bboxes_weighted_feats.shape, rois.shape)
    #     # print("bbox_targets:", bbox_targets)
    #     # neg labels: voc 20
    #     bg_mask = bbox_targets[0] == n_cls
    #     fg_mask = bbox_targets[0] != n_cls
    #
    #     normalize_feat = nn.functional.normalize(bboxes_weighted_feats, dim=0)
    #
    #     norm_feats = norm_tensor(normalize_feat)
    #
    #     norm_neg_queue = center_feats[n_cls]
    #
    #     bg_sim = torch.mm(norm_feats, norm_neg_queue.reshape(-1, 1))
    #
    #     pos_queue = center_feats[:n_cls].permute(1, 0)
    #     # norm_pos_queue = norm_tensor2(pos_queue)
    #     norm_pos_queue = pos_queue
    #     fg_sim = torch.mm(norm_feats, norm_pos_queue)
    #     bg_sim_values, bg_sim_inds = torch.max(bg_sim, dim=1)
    #     fg_sim_values, fg_sim_inds = torch.max(fg_sim, dim=1)
    #
    #     def min_max_norm(x):
    #         if len(x) == 0 or len(x) == 1:
    #             return x
    #         # print("x:", x)
    #         min = torch.min(x, dim=0).values.detach()
    #         max = torch.max(x, dim=0).values.detach()
    #         # print('max:', max, min)
    #         return ((x- min) + eps) / ((max - min) + eps)
    #
    #     bg_cls_weights = torch.mul(torch.clamp(bg_sim_values[bg_mask], min=0.), 1-fg_sim_values[bg_mask])
    #     fg_cls_weights = torch.mul(torch.clamp(fg_sim_values[fg_mask], min=0.), 1-bg_sim_values[fg_mask])
    #
    #     if is_weight_norm is True:
    #         bg_cls_weights = min_max_norm(bg_cls_weights)
    #         fg_cls_weights = min_max_norm(fg_cls_weights)
    #
    #     cls_weights = torch.zeros(len(bbox_results["cls_score"])).to(bbox_results["cls_score"].device).half()
    #     cls_weights = torch.masked_scatter(cls_weights, bg_mask, bg_cls_weights).detach()
    #     cls_weights = torch.masked_scatter(cls_weights, fg_mask, fg_cls_weights).detach()
    #
    #     bbox_weights = cls_weights.reshape((-1, 1)).repeat(1, 4).detach()
    #     loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
    #                                     bbox_results['bbox_pred'],
    #                                     rois,
    #                                     bbox_targets[0],
    #                                     cls_weights,
    #                                     bbox_targets[2],
    #                                     bbox_weights)
    #
    #     bbox_results.update(loss_bbox=loss_bbox)
    #     if return_weights == False:
    #         return bbox_results
    #     else:
    #         return bbox_results, rois, rois_mask, cls_weights, bbox_targets[0]
    #
    # @force_fp32(apply_to=["gt_bboxes", "trans_m"])
    # def _negtive_loss(self,
    #                   x,
    #                   weak_feats,
    #                   trans_m,
    #                   gt_bboxes,
    #                   gt_labels,
    #                   img_metas,
    #                   pos_queue,
    #                   neg_queue,
    #                   weak_imgs=None,
    #                   strong_imgs=None,
    #                   thr=0.001,
    #                   n_cls=20):
    #
    #     strong_bboxes = gt_bboxes
    #     weak_bboxes = self.tran_bbox(gt_bboxes, trans_m, img_metas)
    #
    #     strong_rois = bbox2roi(strong_bboxes).float()
    #     weak_rois = bbox2roi(weak_bboxes).float()
    #
    #     # self.test_weak_bbox(strong_imgs,
    #     #                     weak_imgs,
    #     #                     strong_bboxes,
    #     #                     gt_labels,
    #     #                     weak_bboxes,
    #     #                     trans_m,
    #     #                     img_metas)
    #
    #
    #     bbox_results, strong_feats, _, _ = self._bbox_feat_forward(x, strong_rois)
    #     _, weak_feats, _, _ = self._bbox_feat_forward(weak_feats, weak_rois)
    #
    #     roi_scores = F.softmax(bbox_results['cls_score'], dim=-1)
    #     roi_mask = roi_scores < thr
    #
    #     strong_normalize_feat = norm_tensor(nn.functional.normalize(strong_feats, dim=0))
    #     weak_normalize_feat = norm_tensor(nn.functional.normalize(weak_feats, dim=0))
    #
    #     losses = []
    #     for i in range(len(roi_scores)):
    #         # l_pos: Nx1   l_neg: Nxk
    #         l_pos = torch.mm(strong_normalize_feat[i].reshape(1, -1), weak_normalize_feat[i].reshape(-1, 1)).view((1,-1))
    #         l_pos = l_pos.view((1,)).sum()
    #
    #
    #
    #         if roi_mask[i][n_cls] == True:
    #             neg_keys = torch.cat([pos_queue[roi_mask[i][:n_cls]].permute(1, 0, 2).reshape(neg_queue.shape[0], -1), neg_queue], dim=1)
    #         else:
    #             neg_keys = pos_queue[roi_mask[i][:n_cls]].permute(1, 0, 2).reshape(neg_queue.shape[0], -1)
    #         if neg_keys.shape[1] == 0:
    #             l_neg = torch.tensor(0.).to(x[0].device).view((1,-1))
    #         else:
    #             l_neg = torch.clamp(torch.mm(strong_normalize_feat[i].reshape(1, -1), neg_keys), min=0.).sum() / (neg_keys.shape[1])
    #             # l_neg = torch.mm(strong_normalize_feat[i].reshape(1, -1), neg_keys).view((1,-1))
    #         con_l = 1 - l_pos + torch.sum(l_neg)
    #         losses.append(con_l)
    #
    #     if len(losses) == 0:
    #         losses.append(torch.tensor(0.).to(x[0].device))
    #
    #
    #     return {"unsup_contrast_losses":losses}, roi_mask
    #
    # def _soft_bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, gt_soft_labels,
    #                              img_metas, n_cls=20):
    #
    #     rois = bbox2roi([res.bboxes for res in sampling_results])
    #     def gt_ind(ind_list):
    #         gt_list = []
    #         count=0
    #         for img_id, inds in enumerate(ind_list):
    #             inds = inds + count
    #             gt_list.append(inds)
    #             count = torch.unique(inds).size(0)
    #             # print(count)
    #         rois = torch.cat(gt_list, 0)
    #         return rois
    #
    #     inds = gt_ind([res.pos_assigned_gt_inds for res in sampling_results]).type(torch.long)
    #     bbox_results = self._bbox_forward(x, rois)
    #
    #     bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
    #                                               gt_labels, self.train_cfg)
    #     # print(gt_soft_labels)
    #     # print(bbox_targets[0])
    #     # print(inds)
    #     # print(len(bbox_results['cls_score']), len(inds))
    #     gt_soft_labels = torch.cat(gt_soft_labels, dim=0)
    #     new_bg_labels = torch.zeros(len(bbox_results['cls_score']) - len(inds), n_cls+1).to(inds.device)
    #     new_bg_labels[:, n_cls] = 1.
    #     if len(inds) == 0:
    #         new_labels = new_bg_labels
    #     else:
    #         new_labels = torch.cat([gt_soft_labels[inds], new_bg_labels], dim=0)
    #     # print(type(self.bbox_head))
    #     # print(bbox_results['bbox_pred'])
    #     loss_bbox = self.bbox_head.soft_loss(bbox_results['cls_score'],
    #                                     bbox_results['bbox_pred'], rois,
    #                                         bbox_targets[0],
    #                                     new_labels,
    #                                     bbox_targets[1],
    #                                     bbox_targets[2],
    #                                     bbox_targets[3],)
    #
    #     bbox_results.update(loss_bbox=loss_bbox)
    #     return bbox_results

    # @force_fp32(apply_to=["gt_bboxes", "trans_m"])
    # def _negtive_center_loss(self,
    #                          x,
    #                          weak_feats,
    #                          trans_m,
    #                          gt_bboxes,
    #                          gt_labels,
    #                          img_metas,
    #                          center_feats,
    #                          weak_imgs=None,
    #                          strong_imgs=None,
    #                          thr=0.001,
    #                          n_cls=20):
    #
    #     strong_bboxes = gt_bboxes
    #     weak_bboxes = self.tran_bbox(gt_bboxes, trans_m, img_metas)
    #
    #     strong_rois = bbox2roi(strong_bboxes).float()
    #     weak_rois = bbox2roi(weak_bboxes).float()
    #
    #     # self.test_weak_bbox(strong_imgs,
    #     #                     weak_imgs,
    #     #                     strong_bboxes,
    #     #                     gt_labels,
    #     #                     weak_bboxes,
    #     #                     trans_m,
    #     #                     img_metas)
    #
    #     bbox_results, _, strong_feats, _ = self._bbox_feat_forward(x, strong_rois)
    #     weak_bbox_results, _, weak_feats, _ = self._bbox_feat_forward(weak_feats, weak_rois)
    #
    #
    #     strong_sim = torch.cosine_similarity(strong_feats.unsqueeze(1), center_feats.unsqueeze(0), dim=-1).detach()
    #     weak_sim = torch.cosine_similarity(weak_feats.unsqueeze(1), center_feats.unsqueeze(0), dim=-1).detach()
    #     # roi_masks = (strong_sim < thr) & (weak_sim < thr)
    #
    #     roi_scores = F.softmax(weak_bbox_results['cls_score'], dim=-1)
    #
    #     top_thr = torch.sort(roi_scores, dim=1, descending=True).values[:, thr]
    #     top_thr = top_thr.repeat((n_cls + 1, 1)).T.detach()
    #     roi_masks = roi_scores < top_thr
    #     # print(roi_masks)
    #     strong_normalize_feat = norm_tensor(nn.functional.normalize(strong_feats, dim=0))
    #     weak_normalize_feat = norm_tensor(nn.functional.normalize(weak_feats, dim=0))
    #     # strong_normalize_feat = strong_feats
    #     # weak_normalize_feat = weak_feats
    #     losses = []
    #
    #
    #     # info-nce loss
    #     # if len(strong_normalize_feat) != 0:
    #     #     pos = torch.mm(strong_normalize_feat, weak_normalize_feat.permute(1, 0))
    #     #     neg = torch.mm(strong_normalize_feat, center_feats.permute(1, 0))
    #     #     pos = torch.diag(pos).unsqueeze(-1)
    #     #     # neg = torch.masked_select(neg, roi_masks).reshape(weak_normalize_feat.size(0), -1)
    #     #     neg = torch.masked_fill(neg, ~roi_masks, 0.)
    #     #     # print(neg)
    #     #     # print(pos.shape)
    #     #     # print(neg.shape)
    #     #     N = pos.size(0)
    #     #     logits = torch.cat((pos, neg), dim=1)
    #     #     logits /= 0.4
    #     #     labels = torch.zeros((N,), dtype=torch.long).to(pos.device)
    #     #     con_l = nn.CrossEntropyLoss()(logits, labels)
    #     #     losses.append(con_l)
    #
    #     # cos-sim loss
    #     for i in range(len(roi_masks)):
    #         # l_pos: Nx1   l_neg: Nxk
    #         l_pos = torch.mm(strong_normalize_feat[i].reshape(1, -1), weak_normalize_feat[i].reshape(-1, 1)).view(
    #             (1, -1))
    #         l_pos = l_pos.view((1,)).sum()
    #
    #         # print(center_feats[roi_mask[i]].shape)
    #         neg_keys = center_feats[roi_masks[i]].permute(1, 0)#.reshape(center_feats.shape[0], -1)
    #         if neg_keys.shape[1] == 0:
    #             l_neg = torch.tensor(0.).to(x[0].device).view((1, -1))
    #         else:
    #             l_neg = torch.clamp(torch.mm(strong_normalize_feat[i].reshape(1, -1), neg_keys), min=0.).sum() / (
    #             neg_keys.shape[1])
    #
    #         con_l = 1 - l_pos + torch.sum(l_neg)
    #         con_l = con_l / len(roi_masks)
    #         losses.append(con_l)
    #     if len(losses) == 0:
    #         losses.append(torch.tensor(0.).to(x[0].device))
    #
    #     return {"unsup_contrast_losses": losses}, roi_scores, strong_sim.detach().cpu(), weak_sim.detach().cpu()

    # def _urm_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas, n_cls,
    #                        t_feat=None, teacher=None,
    #                        bboxes=None, bboxes_scores=None, top_thr=3, thr=0.0001,
    #                        is_iou_weight=False, is_neg_loss=False):
    #     rois = bbox2roi([res.bboxes for res in sampling_results])
    #     bbox_results = self._bbox_iou_forward(x, rois)
    #     bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
    #                                               gt_labels, self.train_cfg)
    #
    #     losses = dict()
    #
    #     if is_iou_weight is True:
    #         t_bbox_results = teacher.roi_head._bbox_iou_forward(t_feat, rois)
    #         bbox_weights = torch.sigmoid(t_bbox_results['iou_pred']).repeat(1, 4).detach()
    #         bbox_weights = (bbox_weights > 0.5).float() * bbox_weights
    #         # print(rois)
    #         loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
    #                                         bbox_results['bbox_pred'], rois,
    #                                         bbox_targets[0],
    #                                         bbox_targets[1],
    #                                         bbox_targets[2],
    #                                         bbox_weights)
    #     else:
    #         loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
    #                                         bbox_results['bbox_pred'], rois,
    #                                         *bbox_targets)
    #
    #     losses.update(loss_bbox)
    #     if is_neg_loss is True:
    #         rois = bbox2roi(bboxes).float()
    #         bbox_results = self._bbox_iou_forward(x, rois)
    #         bboxes_scores = torch.cat(bboxes_scores, dim=0)
    #         top_thr = torch.topk(bboxes_scores, k=top_thr, dim=1).values[:, top_thr - 1]
    #         top_thr = top_thr.repeat((n_cls + 1, 1)).T.detach()
    #         roi_masks = bboxes_scores < top_thr
    #         # print(bbox_results['cls_score'].shape, bbox_results['cls_score'])
    #         roi_scores = torch.clamp(F.softmax(bbox_results['cls_score'] * roi_masks, dim=-1), max=0.9999)
    #
    #         # print(torch.log(1-roi_scores*roi_masks))
    #         def reweight(ious):
    #             return 1. / (1. + torch.exp(-10000. * (ious - thr)))
    #
    #         # roi_scores = torch.clamp(F.softmax(bboxes_scores * roi_masks, dim=-1), max=0.9999)
    #         weight = 2 * (1. - (reweight(bboxes_scores * roi_masks) * roi_masks))
    #         neg_losses = -torch.sum(torch.log(1 - roi_scores * roi_masks) * weight) / (torch.sum(roi_masks) + 1)
    #         losses.update(neg_losses)
    #
    #     return

    def _unreliable_neg_loss(self,
                             x,
                             bboxes,
                             bboxes_scores,
                             n_cls,
                             top_thr=5,
                             thr=0.0001):
        rois = bbox2roi(bboxes).float()
        bbox_results = self._bbox_iou_forward(x, rois)
        bboxes_scores = torch.cat(bboxes_scores, dim=0)
        # top_thr = torch.sort(bboxes_scores, dim=1, descending=True).values[:, top_thr]
        top_thr = torch.topk(bboxes_scores, k=top_thr,
                             dim=1).values[:, top_thr-1]
        top_thr = top_thr.repeat((n_cls + 1, 1)).T.detach()
        roi_masks = bboxes_scores < top_thr
        roi_scores = torch.clamp(F.softmax(
            bbox_results['cls_score']*roi_masks, dim=-1), max=0.9999, min=0.0000001)

        roi_scores1 = torch.clamp(F.softmax(
            bbox_results['cls_score'], dim=-1), max=0.9999, min=0.0000001)

        # print(torch.sum(roi_scores*roi_masks).item()/4,
        #       torch.sum(roi_scores1*roi_masks).item()/4)

        def reweight(ious):
            return 1. / (1. + torch.exp(-10000. * (ious - thr)))
        weight = 2 * (1. - (reweight(bboxes_scores*roi_masks) * roi_masks))

        # roi_scores = torch.clamp(F.softmax(bboxes_scores*roi_masks, dim=-1), max=0.9999)
        losses = -torch.sum(torch.log(1-roi_scores*roi_masks)
                            * weight) / (torch.sum(roi_masks) + 1)  # * roi_masks
        # roi_scores = (1 - roi_scores) * roi_masks

        return {"unreliable_neg_loss": losses}, roi_masks, bboxes_scores, torch.sum(roi_scores*roi_masks).item()/4,  torch.sum(roi_scores1*roi_masks).item()/4

    def iou_weighted_loss(self, x, sampling_results, gt_bboxes, gt_labels, img_metas, t_feat, teacher):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_iou_forward(x, rois)
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        t_bbox_results = teacher.roi_head._bbox_iou_forward(t_feat, rois)
        cls_weights = torch.sigmoid(t_bbox_results['iou_pred']).detach()
        cls_weights = (cls_weights > 0.5).float().reshape_as(bbox_targets[1])
        bbox_weights = torch.sigmoid(
            t_bbox_results['iou_pred']).repeat(1, 4).detach()
        bbox_weights = (bbox_weights > 0.5).float()
        # print(bbox_targets[1])
        # print(cls_weights)
        # print(cls_weights.shape,
        #   bbox_results['cls_score'].shape, bbox_targets[1].shape)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        bbox_targets[0],
                                        bbox_targets[1],
                                        bbox_targets[2],
                                        bbox_weights)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results, rois, bbox_weights

    # ========================iou train======================================
    def iou_forward_train(self,
                          x,
                          img_metas,
                          proposal_list,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None,
                          gt_masks=None,
                          **kwargs
                          ):
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            assign_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                assign_results.append(assign_result)
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])
            iou_results = self._sup_iou_loss(x,
                                             sampling_results,
                                             assign_results)
            losses.update(iou_results)
        return losses

    def _bbox_iou_forward(self, x, rois):
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, iou_pred = self.bbox_head.iou_forward(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, iou_pred=iou_pred)
        return bbox_results

    def _sup_iou_loss(self,
                      x,
                      sampling_results,
                      assign_results):
        rois = bbox2roi([res.bboxes for res in sampling_results])

        overlaps_scores = []
        for i in range(len(sampling_results)):
            # print(sampling_results[i].pos_inds)
            overlap_ind = torch.cat(
                [sampling_results[i].pos_inds, sampling_results[i].neg_inds])
            overlaps = assign_results[i].max_overlaps[overlap_ind]
            overlaps_scores.append(overlaps)
            # print(overlaps.shape, overlaps)

        overlaps_scores = torch.cat(overlaps_scores)

        bbox_results = self._bbox_iou_forward(x, rois)

        iou_loss = nn.MSELoss()(torch.sigmoid(
            bbox_results['iou_pred'].reshape(-1)), overlaps_scores)

        return {"iou_losses": iou_loss}

    # ==============================kl loss=================================

    def kl_forward_train(self,
                         x,
                         img_metas,
                         proposal_list,
                         gt_bboxes,
                         gt_labels,
                         gt_bboxes_ignore=None,
                         gt_masks=None,
                         **kwargs
                         ):
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        bbox_results = self.kl_bbox_forward_train(x,
                                                  sampling_results,
                                                  gt_bboxes,
                                                  gt_labels,
                                                  img_metas)
        losses.update(bbox_results['loss_bbox'])
        return losses

    def kl_bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_kl_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        loss_bbox = self.bbox_head.kl_loss(bbox_results['cls_score'],
                                           bbox_results['bbox_pred'],
                                           bbox_results['kl_pred'],
                                           rois,
                                           *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_kl_forward(self, x, rois):
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, kl_pred = self.bbox_head.kl_forward(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, kl_pred=kl_pred)
        return bbox_results

    def tran_bbox(self, bboxes, trans_m, img_metas):
        bboxes = Transform2D.transform_bboxes(bboxes, [m.inverse() for m in trans_m], [
                                              meta["img_shape"] for meta in img_metas])
        return bboxes

    def test_weak_bbox(self,
                       strong_imgs,
                       weak_imgs,
                       strong_bboxes,
                       labels,
                       weak_bboxes,
                       trans_m,
                       img_metas):
        from mmssod.utils.visualization import visual_norm_imgs
        # strong_bboxes = det_bboxes
        # weak_bboxes = self.tran_bbox(strong_bboxes, trans_m, img_metas)

        # print(len(strong_bboxes))
        # print(len(strong_imgs))
        # print(len(img_metas))
        # print(img_metas[0])
        for i in range(len(strong_imgs)):
            visual_norm_imgs(strong_imgs[i], strong_bboxes[i], labels[i], img_metas[i],
                             '/home/liu/ytx/SS-OD/outputs/visual/labels/' + str(self.pic_ind) + '_strong.png')
            visual_norm_imgs(weak_imgs[i], weak_bboxes[i], labels[i], img_metas[i],
                             '/home/liu/ytx/SS-OD/outputs/visual/labels/' + str(self.pic_ind) + '_weak.png')
            self.pic_ind += 1
