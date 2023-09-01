from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from mmdet.core import multi_apply
import torch
from torch import nn
from mmcv.cnn import build_norm_layer

@HEADS.register_module()
class Shared2FCFeatBBoxHead(Shared2FCBBoxHead):
    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCFeatBBoxHead, self).__init__(
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.projector = nn.Sequential(
            nn.Linear(fc_out_channels, fc_out_channels),
            # build_norm_layer(dict(type='BN'), num_features=fc_out_channels)[1],
            nn.BatchNorm1d(fc_out_channels),
            # nn.BatchNorm2d(fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(fc_out_channels, fc_out_channels))


    def forward_feats(self, x):
        # shared part
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

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)

        projector_feats = self.projector(x_cls)

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
        return cls_score, bbox_pred, x_cls, projector_feats
