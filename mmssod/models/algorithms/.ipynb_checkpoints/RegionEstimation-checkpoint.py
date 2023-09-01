import torch
from mmdet.models import DETECTORS
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
class RegionEstimation(BurnInTSModel):
    def __init__(self, teacher: dict, student: dict, train_cfg=None, test_cfg=None, n_cls=20):
        super().__init__(teacher, student, train_cfg, test_cfg, n_cls)
        print("train_cfg:",train_cfg)

        self.train_cfg = train_cfg


        assigner_cfg = train_cfg.get("assigner", None)
        self.is_assigner_recreate = False
        if assigner_cfg is not None:
            self.is_assigner_recreate = True
            self.rcnn_bbox_unsup_assigner = build_assigner(train_cfg.assigner)
            self.rcnn_bbox_sup_assigner = build_assigner(self.student.train_cfg.rcnn.assigner)

        # queues settings
        self.feat_dim = train_cfg.get("feat_dim", 1024)
        self.pos_queue_len = train_cfg.get("pos_queue_len", 100)
        self.neg_queue_len = train_cfg.get("neg_queue_len", 65536)
        self.region_bg_max_num = train_cfg.get("region_bg_max_num", 10)
        self.region_fg_max_num = train_cfg.get("region_fg_max_num", 80)

        # store the keys of pos/neg regions
        self.register_buffer('pos_queue', torch.zeros(self.n_cls ,self.feat_dim, self.pos_queue_len))
        self.register_buffer('pos_queue_ptr', torch.zeros((self.n_cls, 1), dtype=torch.long))
        self.register_buffer('neg_queue', torch.zeros(self.feat_dim, self.neg_queue_len))
        self.register_buffer('neg_queue_ptr', torch.zeros(1, dtype=torch.long))

        self.is_unreliable = train_cfg.get("unreliable", False)
        self.unreliable_thr = train_cfg.get("unreliable_thr", 0.7)

        if self.is_unreliable is True:
            self.unreliable_pos_queue_len = train_cfg.get("unreliable_pos_queue_len", 1000) * self.n_cls
            self.unreliable_neg_queue_len = train_cfg.get("unreliable_neg_queue_len", 65536)

            self.register_buffer('unreliable_pos_queue', torch.zeros(self.feat_dim, self.unreliable_pos_queue_len))
            self.register_buffer('unreliable_pos_queue_ptr', torch.zeros(1, dtype=torch.long))
            self.register_buffer('unreliable_neg_queue', torch.zeros(self.feat_dim, self.unreliable_neg_queue_len))
            self.register_buffer('unreliable_neg_queue_ptr', torch.zeros(1, dtype=torch.long))
        else:
            self.unreliable_pos_queue = None
            self.unreliable_neg_queue = None




    def forward_train(self, imgs, img_metas, **kwargs):
        kwargs.update({"img": imgs})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")

        losses = dict()
        if self.is_assigner_recreate is True:
            self._switch_sup_train()
        sup_loss = self.student.forward_train(**data_groups["sup"])
        sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
        losses.update(sup_loss)

        self._update_reliable_queues(data_groups["sup"])
        strong_unsup = self._gen_pseudo_labels(data_groups["unsup"])
        if self.is_unreliable is True:
            self._update_unreliable_queues(strong_unsup)

        if self.is_assigner_recreate is True:
            self._switch_unsup_train()
        unsup_loss = self._compute_student_unsup_losses(strong_unsup)
        unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
        losses.update(unsup_loss)

        losses = weighted_loss(losses, self.unsup_loss_weight)
        return losses



    def _gen_pseudo_labels(self, unsup_data):
        self.teacher.eval()
        weak_unsup, strong_unsup = self._split_unsup_data(unsup_data)

        # extract feats
        feat = self.teacher.extract_feat(weak_unsup['img'])

        # extract proposal regions
        proposal_list = self.teacher.rpn_head.simple_test_rpn(feat, weak_unsup['img_metas'])

        # extract det results from the detector
        det_bboxes, det_labels = self.teacher.roi_head.simple_test_bboxes(
            feat, weak_unsup['img_metas'], proposal_list, self.teacher.test_cfg.rcnn, rescale=False)

        result_bboxes = [bbox[:, :4] for bbox in det_bboxes]
        result_scores = [bbox[:, 4] for bbox in det_bboxes]

        # filter bboxes using thr
        gt_bboxes = [bboxes[scores >= self.pesudo_thr] for (bboxes, scores) in zip(result_bboxes, result_scores)]
        gt_labels = [labels[scores >= self.pesudo_thr] for (labels, scores) in zip(det_labels, result_scores)]
        gt_scores = [scores1[scores2 >= self.pesudo_thr] for (scores1, scores2) in zip(result_scores, result_scores)]

        # transform det bbox
        M = self._extract_transform_matrix(weak_unsup, strong_unsup)
        gt_bboxes = self._transform_bbox(
            gt_bboxes,
            M,
            [meta["img_shape"] for meta in strong_unsup["img_metas"]],
        )



        if self.is_unreliable is True:
            unreliable_bboxes = [bboxes[scores >= self.unreliable_thr] for (bboxes, scores) in zip(result_bboxes, result_scores)]
            unreliable_labels = [labels[scores >= self.unreliable_thr] for (labels, scores) in zip(det_labels, result_scores)]

            # transform det bbox
            M = self._extract_transform_matrix(weak_unsup, strong_unsup)
            unreliable_bboxes = self._transform_bbox(
                unreliable_bboxes,
                M,
                [meta["img_shape"] for meta in strong_unsup["img_metas"]],
            )
            strong_unsup.update({"unreliable_bboxes": unreliable_bboxes})
            strong_unsup.update({"unreliable_labels": unreliable_labels})



        # replace with the pesudo labels/bboxes
        if self.print_pesudo_summary:
            self._add_summary_bboxes([{"bboxes":torch.cat([bboxes.detach().cpu().clone(), torch.unsqueeze(scores.clone(), dim=1).detach().cpu().clone()], dim=1).detach().cpu().numpy(), "labels":labels.detach().cpu().numpy()} for (bboxes, scores, labels) in zip(gt_bboxes, gt_scores, gt_labels)],
                                     [{"bboxes":bboxes.detach().cpu().numpy(), "labels":labels.detach().cpu().numpy()} for (bboxes, labels) in zip(strong_unsup["gt_bboxes"], strong_unsup["gt_labels"])])
            strong_unsup.update({"gt_bboxes_true": [bboxes for bboxes in strong_unsup["gt_bboxes"]]})
            strong_unsup.update({"gt_labels_true": [labels for labels in strong_unsup["gt_labels"]]})

        strong_unsup.update({"gt_bboxes" : gt_bboxes})
        strong_unsup.update({"gt_labels" : gt_labels})

        return strong_unsup

    @torch.no_grad()
    def _update_reliable_queues(self, sup_data, proposals=None):
        self.teacher.eval()
        feat = self.teacher.extract_feat(sup_data['img'])
        proposal_list = self.teacher.rpn_head.simple_test_rpn(feat, sup_data['img_metas'])

        bg_feats, gt_feats = self.teacher.roi_head.extract_roi_features(
            feat, proposal_list, sup_data["gt_bboxes"] ,self.train_cfg, self.train_cfg.region_bg_score_thr)

        bg_feats = norm_tensor(nn.functional.normalize(bg_feats, dim=0))
        gt_feats = norm_tensor(nn.functional.normalize(gt_feats, dim=0))

        self._neg_dequeue_and_enqueue(bg_feats, img_num=len(proposal_list))
        self._pos_dequeue_and_enqueue(gt_feats, torch.cat(sup_data["gt_labels"], dim=0), img_num=len(proposal_list))

    @torch.no_grad()
    def _update_unreliable_queues(self, unsup_data, proposals=None):
        self.teacher.eval()
        feat = self.teacher.extract_feat(unsup_data['img'])
        proposal_list = self.teacher.rpn_head.simple_test_rpn(feat, unsup_data['img_metas'])

        bg_feats, gt_feats = self.teacher.roi_head.extract_roi_features(
            feat, proposal_list, unsup_data["unreliable_bboxes"], self.train_cfg, self.train_cfg.unreliable_region_bg_score_thr)

        # bg_feats = norm_tensor(nn.functional.normalize(bg_feats, dim=0))
        gt_feats = norm_tensor(nn.functional.normalize(gt_feats, dim=0))

        # bg_feats = nn.functional.normalize(bg_feats, dim=0)
        # gt_feats = nn.functional.normalize(gt_feats, dim=0)

        # self._unreliable_neg_dequeue_and_enqueue(bg_feats, img_num=len(proposal_list))
        self._unreliable_pos_dequeue_and_enqueue(gt_feats, img_num=len(proposal_list))

    @torch.no_grad()
    def _neg_dequeue_and_enqueue(self, keys, img_num=0):
        """Update neg region queue."""

        broadcast_keys = torch.zeros(img_num * self.region_bg_max_num, self.feat_dim).to(keys.device)
        broadcast_mask = torch.zeros(img_num * self.region_bg_max_num).to(keys.device)
        # broadcast_mask[:]
        broadcast_keys[:len(keys)] = keys
        broadcast_mask[:len(keys)] = 1.0
        # print(broadcast_mask)
        bg_keys = gather_same_shape_tensors(broadcast_keys)
        bg_masks = gather_same_shape_tensors(broadcast_mask)
        keys = bg_keys[bg_masks == 1.0]

        update_size = keys.shape[0]
        ptr = int(self.neg_queue_ptr)

        if (ptr + update_size) > self.neg_queue_len:
            len_11 = self.neg_queue_len-ptr
            self.neg_queue[:, ptr:] = keys[:len_11].transpose(0, 1)
            ptr = (ptr + update_size) % self.neg_queue_len
            self.neg_queue[:, :ptr] = keys[len_11:].transpose(0, 1)
        else:
            self.neg_queue[:, ptr:ptr + update_size] = keys.transpose(0, 1)
            ptr = (ptr + update_size) % self.neg_queue_len  # move pointer
        #
        self.neg_queue_ptr[0] = ptr

    @torch.no_grad()
    def _pos_dequeue_and_enqueue(self, keys, gt_labels, img_num=0):
        """Update pos region queue."""
        broadcast_keys = torch.zeros(img_num * self.region_fg_max_num, self.feat_dim).to(keys.device)
        broadcast_labels = torch.zeros(img_num * self.region_fg_max_num).to(keys.device)
        broadcast_mask = torch.zeros(img_num * self.region_fg_max_num).to(keys.device)
        # broadcast_mask[:]
        broadcast_keys[:len(keys)] = keys
        broadcast_labels[:len(keys)] = gt_labels
        broadcast_mask[:len(keys)] = 1.0
        # print(broadcast_mask)
        fg_keys = gather_same_shape_tensors(broadcast_keys)
        fg_masks = gather_same_shape_tensors(broadcast_mask)
        fg_labels = gather_same_shape_tensors(broadcast_labels)
        keys = fg_keys[fg_masks == 1.0]
        gts_labels = fg_labels[fg_masks == 1.0]


        for i in range(self.n_cls):
            update_size = keys[gts_labels==i].shape[0]
            ptr = int(self.pos_queue_ptr[i])

            if (ptr + update_size) > self.pos_queue_len:
                len_11 = self.pos_queue_len - ptr
                self.pos_queue[i, :, ptr:] = keys[gts_labels==i][:len_11].transpose(0, 1)
                ptr = (ptr + update_size) % self.pos_queue_len
                self.pos_queue[i, :, :ptr] = keys[gts_labels==i][len_11:].transpose(0, 1)
            else:
                self.pos_queue[i, :, ptr:ptr + update_size] = keys[gts_labels==i].transpose(0, 1)
                ptr = (ptr + update_size) % self.pos_queue_len  # move pointer
            #
            self.pos_queue_ptr[i][0] = ptr

    def _unreliable_neg_dequeue_and_enqueue(self, keys, img_num=0):
        broadcast_keys = torch.zeros(img_num * self.region_bg_max_num, self.feat_dim).to(keys.device)
        broadcast_mask = torch.zeros(img_num * self.region_bg_max_num).to(keys.device)
        # broadcast_mask[:]
        broadcast_keys[:len(keys)] = keys
        broadcast_mask[:len(keys)] = 1.0
        # print(broadcast_mask)
        bg_keys = gather_same_shape_tensors(broadcast_keys)
        bg_masks = gather_same_shape_tensors(broadcast_mask)
        keys = bg_keys[bg_masks == 1.0]

        update_size = keys.shape[0]
        ptr = int(self.unreliable_neg_queue_ptr)

        if (ptr + update_size) > self.unreliable_neg_queue_len:
            len_11 = self.unreliable_neg_queue_len - ptr
            self.unreliable_neg_queue[:, ptr:] = keys[:len_11].transpose(0, 1)
            ptr = (ptr + update_size) % self.unreliable_neg_queue_len
            self.unreliable_neg_queue[:, :ptr] = keys[len_11:].transpose(0, 1)
        else:
            self.unreliable_neg_queue[:, ptr:ptr + update_size] = keys.transpose(0, 1)
            ptr = (ptr + update_size) % self.unreliable_neg_queue_len  # move pointer
        #
        self.unreliable_neg_queue_ptr[0] = ptr

    def _unreliable_pos_dequeue_and_enqueue(self, keys, img_num=0):
        broadcast_keys = torch.zeros(img_num * self.region_fg_max_num, self.feat_dim).to(keys.device)
        broadcast_mask = torch.zeros(img_num * self.region_fg_max_num).to(keys.device)
        # broadcast_mask[:]
        broadcast_keys[:len(keys)] = keys
        broadcast_mask[:len(keys)] = 1.0
        # print(broadcast_mask)
        fg_keys = gather_same_shape_tensors(broadcast_keys)
        fg_masks = gather_same_shape_tensors(broadcast_mask)
        keys = fg_keys[fg_masks == 1.0]

        update_size = keys.shape[0]
        ptr = int(self.unreliable_pos_queue_ptr)

        if (ptr + update_size) > self.unreliable_pos_queue_len:
            len_11 = self.unreliable_pos_queue_len - ptr
            self.unreliable_pos_queue[:, ptr:] = keys[:len_11].transpose(0, 1)
            ptr = (ptr + update_size) % self.unreliable_pos_queue_len
            self.unreliable_pos_queue[:, :ptr] = keys[len_11:].transpose(0, 1)
        else:
            self.unreliable_pos_queue[:, ptr:ptr + update_size] = keys.transpose(0, 1)
            ptr = (ptr + update_size) % self.unreliable_pos_queue_len  # move pointer
        #
        self.unreliable_pos_queue_ptr[0] = ptr

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
                sampling_results.append(sampling_result)

        losses = dict()
        if self.student.roi_head.with_bbox:
            bbox_results, rois, rois_mask, label_weights = self.student.roi_head._weighted_bbox_forward_train(x,
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
            if self.print_pesudo_summary:
                for idx_img in range(num_imgs):
                    mask = (rois[:, 0] == idx_img)
                    pos_bboxes = rois[mask][rois_mask[mask] == 1.0]
                    neg_bboxes = rois[mask][rois_mask[mask] == 0.0]

                    pos_weight = label_weights[mask][rois_mask[mask] == 1.0]
                    neg_weight = label_weights[mask][rois_mask[mask] == 0.0]

                    self._add_unsup_sampling_bboxes(pos_bboxes.detach().cpu(),
                                                    neg_bboxes.detach().cpu(),
                                                    gt_bboxes_true[idx_img].detach().cpu(),
                                                    pos_weight.detach().cpu(),
                                                    neg_weight.detach().cpu())
            losses.update(bbox_results['loss_bbox'])
            if self.is_unreliable is True:
                losses.update(unreliable_loss=bbox_results['unreliable_loss'])

        return losses

    def _switch_sup_train(self):
        self.student.roi_head.bbox_assigner = self.rcnn_bbox_sup_assigner

    def _switch_unsup_train(self):
        self.student.roi_head.bbox_assigner = self.rcnn_bbox_unsup_assigner
