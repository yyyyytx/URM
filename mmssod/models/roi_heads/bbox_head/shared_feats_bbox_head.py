from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from mmdet.core import multi_apply
import torch
from torch import nn
from mmcv.cnn import build_norm_layer
from mmdet.models.utils import build_linear_layer
from mmcv.runner import BaseModule, auto_fp16, force_fp32

@HEADS.register_module()
class Shared2FCFeatBBoxHead(Shared2FCBBoxHead):
    def __init__(self, fc_out_channels=1024, n_cls=20, *args, **kwargs):
        super(Shared2FCFeatBBoxHead, self).__init__(
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        print("num:", self.num_classes)
        self.fc_iou = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=1)

        # out_dim_reg = (4 if self.reg_class_agnostic else 4 *
        #                                                  self.num_classes)
        # self.kl_fc = build_linear_layer(
        #     self.reg_predictor_cfg,
        #     in_features=self.reg_last_dim,
        #     out_features=out_dim_reg)

        # self.mlp_fc1 = nn.Linear(self.in_channels * self.roi_feat_area, fc_out_channels)
        # self.mlp_bn = nn.BatchNorm1d(fc_out_channels)
        # self.mlp_fc2 = nn.Linear(fc_out_channels, 128)

    def kl_forward(self, x):
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        x_kl = x

        # faster rcnn:  self.cls_convs is []
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)

        # faster rcnn:  self.cls_fcs is []
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        kl_pred = self.kl_fc(x_kl)
        return cls_score, bbox_pred, kl_pred

    def iou_forward(self,x):
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        # faster rcnn:  self.cls_convs is []
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)

        # faster rcnn:  self.cls_fcs is []
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        iou_pred = self.fc_iou(x.detach())
        return cls_score, bbox_pred, iou_pred


    def forward_feats(self, x):
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)
            # projector head
            # projector_feats = self.mlp_fc1(x)
            # if projector_feats.shape[0] > 1:
            #     projector_feats = self.mlp_bn(projector_feats)
            # projector_feats = self.relu(projector_feats)
            # projector_feats = self.mlp_fc2(projector_feats)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        # faster rcnn:  self.cls_convs is []
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)




        # faster rcnn:  self.cls_fcs is []
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        iou_pred = self.fc_iou(x.detach())
        return cls_score, bbox_pred, None, x, iou_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def soft_loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             soft_labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    soft_labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():

                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), -1,
                    4)[pos_inds.type(torch.bool),
                       labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'pred_stds'))
    def kl_loss(self,
                cls_score,
                bbox_pred,
                pred_stds,
                rois,
                labels,
                label_weights,
                bbox_targets,
                bbox_weights,
                reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_

        # if bbox_pred is not None:
        #     bg_class_ind = self.num_classes
        #     # 0~self.num_classes-1 are FG, self.num_classes is BG
        #     pos_inds = (labels >= 0) & (labels < bg_class_ind)
        #     # do not perform bounding box regression for BG anymore.
        #     if pos_inds.any():
        #         pos_bbox_pred = bbox_pred.view(
        #             bbox_pred.size(0), -1,
        #             4)[pos_inds.type(torch.bool),
        #                labels[pos_inds.type(torch.bool)]]
        #
        #         losses['loss_bbox'] = self.loss_bbox(
        #             pos_bbox_pred,
        #             bbox_targets[pos_inds.type(torch.bool)],
        #             bbox_weights[pos_inds.type(torch.bool)],
        #             avg_factor=bbox_targets.size(0),
        #             reduction_override=reduction_override)
        #     else:
        #         losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        if pred_stds is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), -1,
                    4)[pos_inds.type(torch.bool),
                       labels[pos_inds.type(torch.bool)]]
                pred_stds = pred_stds.view(bbox_pred.size(0), -1,
                    4)[pos_inds.type(torch.bool),
                       labels[pos_inds.type(torch.bool)]]

                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)

                # alpha = torch.log(pred_stds*pred_stds)
                alpha = pred_stds
                item1_weight = torch.square(pos_bbox_pred - bbox_targets[pos_inds.type(torch.bool)]).detach()
                item1 = torch.exp(-alpha)
                item2 = alpha
                # print(item1)
                # print(alpha)
                kl_loss = torch.mean(0.5 * (item2 + item1 * item1_weight))
                # print(kl_loss)
                losses['kl_loss'] = kl_loss * 0.5

            else:
                losses['kl_loss'] = bbox_pred[pos_inds].sum()

        return losses