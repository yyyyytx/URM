from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead
import torch
from mmdet.core import (bbox2roi)
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

    def _bbox_feat_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, projector_feats, feats = self.bbox_head.forward_feats(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results, projector_feats, feats

    def extract_projector_features(self,
                                   x,
                                   proposals,
                                   gt_bboxes,
                                   train_cfg,
                                   bg_score_thr=0.3):
        bg_roi_list = []
        for i in range(len(proposals)):
            rois = proposals[i][:, :4]
            roi_scores = proposals[i][:, 4]
            overlaps, _ = cal_bboxes_overlaps(rois, gt_bboxes[i], mode='iof')
            rois = rois[overlaps < train_cfg.region_bg_iou_thr]
            roi_scores = roi_scores[overlaps < train_cfg.region_bg_iou_thr].view(-1, 1).repeat(1, 2)
            bg_rois, _ = multiclass_nms(rois,
                                        roi_scores,
                                        bg_score_thr,
                                        train_cfg.region_bg_nms_cfg,
                                        max_num=train_cfg.region_bg_max_num)
            img_inds = bg_rois.new_full((bg_rois.size(0), 1), i)
            bg_rois = torch.cat([img_inds, bg_rois[:, :4]], dim=-1)

            bg_roi_list.append(bg_rois)
        bg_rois = torch.cat(bg_roi_list, 0)
        _, bg_projector_feats, bg_feats = self._bbox_feat_forward(x, bg_rois)

        gt_roi_list = []
        for i in range(len(gt_bboxes)):
            gt_rois = gt_bboxes[i]
            img_inds = gt_rois.new_full((gt_rois.size(0), 1), i)
            gt_rois = torch.cat([img_inds, gt_rois[:, :4]], dim=-1)

            gt_roi_list.append(gt_rois)
        gt_rois = torch.cat(gt_roi_list, 0)
        _, gt_projector_feats, gt_feats = self._bbox_feat_forward(x, gt_rois)
        return bg_projector_feats, gt_projector_feats, bg_feats, gt_feats

    def _weighted_bbox_forward_train(self,
                                     x,
                                     sampling_results,
                                     gt_bboxes,
                                     gt_labels,
                                     pos_queue,
                                     neg_queue,
                                     n_cls=20,
                                     return_weights=False):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        rois_mask = bbox2mask([(len(res.pos_bboxes), len(res.neg_bboxes)) for res in sampling_results])

        bbox_results, bboxes_weighted_feats, _ = self._bbox_feat_forward(x, rois)
        # print("bbox_results:", bbox_results['cls_score'].shape, bbox_results['bbox_pred'].shape, bboxes_weighted_feats.shape)
        # print(bbox_results['cls_score'])

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        # print(bboxes_weighted_feats.shape, rois.shape)
        # print("bbox_targets:", bbox_targets)
        # neg labels: voc 20
        bg_mask = bbox_targets[0] == n_cls
        fg_mask = bbox_targets[0] != n_cls

        normalize_feat = nn.functional.normalize(bboxes_weighted_feats, dim=0)

        norm_feats = norm_tensor(normalize_feat)
        # norm_neg_queue = norm_tensor2(neg_queue)
        norm_neg_queue = neg_queue
        bg_sim = torch.mm(norm_feats, norm_neg_queue)

        pos_queue = pos_queue.permute(1, 0, 2).reshape(norm_feats.shape[1], -1)
        # norm_pos_queue = norm_tensor2(pos_queue)
        norm_pos_queue = pos_queue
        fg_sim = torch.mm(norm_feats, norm_pos_queue)
        bg_sim_values, bg_sim_inds = torch.max(bg_sim, dim=1)
        fg_sim_values, fg_sim_inds = torch.max(fg_sim, dim=1)

        # def min_max_norm(x):
        #     if len(x) == 0 or len(x) == 1:
        #         return x
        #     # print("x:", x)
        #     min = torch.min(x, dim=0).values.detach()
        #     max = torch.max(x, dim=0).values.detach()
        #     # print('max:', max, min)
        #     return ((x- min) + eps) / ((max - min) + eps)

        label_weights = torch.zeros(len(bbox_results["cls_score"])).to(bbox_results["cls_score"].device)
        bg_cls_weights = torch.mul(torch.clamp(bg_sim_values[bg_mask], min=0.), 1-fg_sim_values[bg_mask])
        fg_cls_weights = torch.mul(torch.clamp(fg_sim_values[fg_mask], min=0.), 1-bg_sim_values[fg_mask])
        # print("before:", fg_cls_weights)
        # print("before:", fg_cls_weights)

        # bg_cls_weights = min_max_norm(bg_cls_weights)
        # fg_cls_weights = min_max_norm(fg_cls_weights)
        # print("after:", fg_cls_weights)
        # print("after:", fg_cls_weights)

        cls_weights = torch.masked_scatter(label_weights.half(), bg_mask, bg_cls_weights).detach()
        cls_weights = torch.masked_scatter(cls_weights.half(), fg_mask, fg_cls_weights).detach()
        # print("before:", cls_weights)

        # cls_weights = min_max_norm(cls_weights)
        # print("after:", cls_weights)

        # print(cls_weights)

        bbox_weights = cls_weights.reshape((-1, 1)).repeat(1, 4).detach()
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'],
                                        rois,
                                        bbox_targets[0],
                                        cls_weights,
                                        bbox_targets[2],
                                        bbox_weights)

        bbox_results.update(loss_bbox=loss_bbox)
        if return_weights == False:
            return bbox_results
        else:
            return bbox_results, rois, rois_mask, cls_weights, bbox_targets[0]

    @force_fp32(apply_to=["gt_bboxes", "trans_m"])
    def _negtive_loss(self,
                      x,
                      weak_feats,
                      trans_m,
                      gt_bboxes,
                      gt_labels,
                      img_metas,
                      pos_queue,
                      neg_queue,
                      weak_imgs=None,
                      strong_imgs=None,
                      n_cls=20):

        strong_bboxes = gt_bboxes
        weak_bboxes = self.tran_bbox(gt_bboxes, trans_m, img_metas)

        strong_rois = bbox2roi(strong_bboxes).float()
        weak_rois = bbox2roi(weak_bboxes).float()

        # self.test_weak_bbox(strong_imgs,
        #                     weak_imgs,
        #                     strong_bboxes,
        #                     gt_labels,
        #                     weak_bboxes,
        #                     trans_m,
        #                     img_metas)


        bbox_results, strong_feats, _ = self._bbox_feat_forward(x, strong_rois)
        _, weak_feats, _ = self._bbox_feat_forward(weak_feats, weak_rois)

        roi_scores = F.softmax(bbox_results['cls_score'], dim=-1)
        roi_mask = roi_scores < 0.001
        # mask_labels = torch.cat(gt_labels).reshape(-1, 1)
        # roi_mask = torch.scatter(roi_mask, 1, mask_labels, False)

        strong_normalize_feat = norm_tensor(nn.functional.normalize(strong_feats, dim=0))
        weak_normalize_feat = norm_tensor(nn.functional.normalize(weak_feats, dim=0))

        losses = []
        for i in range(len(roi_scores)):
            # l_pos: Nx1   l_neg: Nxk
            l_pos = torch.mm(strong_normalize_feat[i].reshape(1, -1), weak_normalize_feat[i].reshape(-1, 1)).view((1,-1))
            l_pos = l_pos.view((1,)).sum()



            if roi_mask[i][n_cls] == True:
                neg_keys = torch.cat([pos_queue[roi_mask[i][:n_cls]].permute(1, 0, 2).reshape(neg_queue.shape[0], -1), neg_queue], dim=1)
            else:
                neg_keys = pos_queue[roi_mask[i][:n_cls]].permute(1, 0, 2).reshape(neg_queue.shape[0], -1)
            if neg_keys.shape[1] == 0:
                l_neg = torch.tensor(0.).to(x[0].device).view((1,-1))
            else:
                l_neg = torch.clamp(torch.mm(strong_normalize_feat[i].reshape(1, -1), neg_keys), min=0.).sum() / (neg_keys.shape[1])
                # l_neg = torch.mm(strong_normalize_feat[i].reshape(1, -1), neg_keys).view((1,-1))
            con_l = 1 - l_pos + torch.sum(l_neg)
            losses.append(con_l)

            # N = l_pos.size(0)
            # print(l_pos.shape, l_neg.shape)
            # logits = torch.cat((l_pos, l_neg), dim=1)
            # logits /= 0.2
            # labels = torch.zeros((N,), dtype=torch.long).to(l_pos.device)
            # con_l = nn.CrossEntropyLoss()(logits, labels)
        if len(losses) == 0:
            losses.append(torch.tensor(0.).to(x[0].device))


        return {"unsup_contrast_losses":losses}, roi_mask

    @force_fp32(apply_to=["gt_bboxes", "trans_m"])
    def _sup_negtive_loss(self,
                          x,
                          sampling_results,
                          gt_bboxes,
                          gt_labels,
                          pos_queue,
                          neg_queue,
                          n_cls=20):

        rois = bbox2roi([res.bboxes for res in sampling_results])

        bbox_results, projector_bboxes_feats, _ = self._bbox_feat_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        bg_mask = bbox_targets[0] == n_cls
        fg_mask = bbox_targets[0] != n_cls

        normalize_feat = nn.functional.normalize(projector_bboxes_feats, dim=0)
        norm_feats = norm_tensor(normalize_feat)


        pos_queue = pos_queue.permute(1, 0, 2).reshape(norm_feats.shape[1], -1)
        norm_pos_queue = pos_queue
        norm_neg_queue = neg_queue

        # losses = []
        l_fg_neg = torch.clamp(torch.mm(projector_bboxes_feats[fg_mask], norm_neg_queue), min=0.)
        l_fg_neg = l_fg_neg[l_fg_neg > 0.]
        l_bg_neg = torch.clamp(torch.mm(projector_bboxes_feats[bg_mask], norm_pos_queue), min=0.)
        l_bg_neg = l_bg_neg[l_bg_neg > 0.]
        # print(torch.sum(l_fg_neg), torch.sum(l_bg_neg))
        neg_loss = (torch.sum(l_fg_neg) + torch.sum(l_bg_neg)) / (len(l_fg_neg) + len(l_bg_neg) + 1)
        # l_fg_neg = torch.mean(torch.clamp(torch.mm(projector_bboxes_feats[fg_mask], norm_neg_queue), min=0.))
        # l_bg_neg = torch.mean(torch.clamp(torch.mm(projector_bboxes_feats[bg_mask], norm_pos_queue), min=0.))
        # neg_loss = l_fg_neg + l_bg_neg

        return {"sup_contrast_losses": neg_loss}

    def tran_bbox(self, bboxes, trans_m, img_metas):
        bboxes = Transform2D.transform_bboxes(bboxes, [m.inverse() for m in trans_m], [meta["img_shape"] for meta in img_metas])
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

        print(len(strong_bboxes))
        print(len(strong_imgs))
        print(len(img_metas))
        print(img_metas[0])
        for i in range(len(strong_imgs)):
            visual_norm_imgs(strong_imgs[i], strong_bboxes[i], labels[i], img_metas[i], '/home/liu/ytx/SS-OD/outputs/visual/labels/' + str(self.pic_ind) + '_strong.png')
            visual_norm_imgs(weak_imgs[i], weak_bboxes[i], labels[i], img_metas[i], '/home/liu/ytx/SS-OD/outputs/visual/labels/' + str(self.pic_ind) + '_weak.png')
            self.pic_ind += 1