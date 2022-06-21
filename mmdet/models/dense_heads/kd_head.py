# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import AbstractContextManager
from tkinter import W
from typing import Dict, List, Tuple

import einops
import todd
import torch.nn.functional as F
from mmcv.runner import ModuleDict

from ..builder import HEADS, build_head
from .base_dense_head import BaseDenseHead
from .fcos_head import FCOSHead
from .retina_head import RetinaHead
from .rpn_head import RPNHead


class WithPos(AbstractContextManager):

    def __init__(self, prior_generator):
        self._prior_generator = prior_generator

    def __enter__(self):
        self._prev = self._prior_generator.with_pos
        self._prior_generator.with_pos = True

    def __exit__(self, __exc_type, __exc_value, __traceback):
        self._prior_generator.with_pos = self._prev


class MultiHeadMixin:
    """Single stage head with extra parallel heads."""

    def __init__(self,
                 *args,
                 extra_heads: Dict[str, dict] = None,
                 pool: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        extra_heads = {} if extra_heads is None else extra_heads
        self._extra_heads: Dict[str, BaseDenseHead] = ModuleDict(
            {name: build_head(cfg)
             for name, cfg in extra_heads.items()})
        self._pool = pool

    def forward_train(
        self,
        x,
        img_metas,
        gt_bboxes,
        gt_labels=None,
        gt_bboxes_ignore=None,
        proposal_cfg=None,
        **kwargs,
    ):
        outs = super().forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            proposal_cfg,
            **kwargs,
        )

        losses = outs if proposal_cfg is None else outs[0]
        if self._pool:
            x = [F.avg_pool2d(feat, 2) for feat in x]
        for head_name, head in self._extra_heads.items():
            head_losses = head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
                None,
                **kwargs,
            )
            losses.update({
                f'{loss_name}_{head_name}': loss
                for loss_name, loss in head_losses.items()
            })

        return outs


@HEADS.register_module()
class RPNHead_(RPNHead):

    def __init__(self,
                 *args,
                 cache_anchors: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_anchors = cache_anchors

    def get_anchors(self, featmap_sizes: List[Tuple[int]], *args, **kwargs):
        anchor_list, valid_flag_list = super().get_anchors(
            featmap_sizes, *args, **kwargs)
        if self._cache_anchors:
            self.anchors = [
                anchors.reshape(featmap_size + (3, 4))
                for featmap_size, anchors in zip(featmap_sizes, anchor_list[0])
            ]
        return anchor_list, valid_flag_list


@HEADS.register_module()
class RetinaMultiHead(MultiHeadMixin, RetinaHead):

    def __init__(self,
                 *args,
                 cache_feat_mask: bool = False,
                 cache_anchors: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_feat_mask = cache_feat_mask
        self._cache_anchors = cache_anchors

    def get_anchors(self, featmap_sizes: List[Tuple[int]], *args, **kwargs):
        anchor_list, valid_flag_list = super().get_anchors(
            featmap_sizes, *args, **kwargs)
        if self._cache_anchors:
            self.anchors = [
                anchors.reshape(featmap_size + (9, 4))
                for featmap_size, anchors in zip(featmap_sizes, anchor_list[0])
            ]
        return anchor_list, valid_flag_list

    def loss_single(
        self,
        cls_score,
        bbox_pred,
        anchors,
        labels,
        label_weights,
        bbox_targets,
        bbox_weights,
        num_total_samples,
    ):
        if self._cache_feat_mask:
            bs, _, feat_h, feat_w = cls_score.shape
            labels2d = labels.reshape(bs, feat_h, feat_w,
                                      -1) != self.num_classes
            feat_mask, _ = labels2d.max(-1)
            self.feat_mask.append(feat_mask)
        return super().loss_single(
            cls_score,
            bbox_pred,
            anchors,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            num_total_samples,
        )

    def loss(self, *args, **kwargs):
        if self._cache_feat_mask:
            self.feat_mask = []
        losses = super().loss(*args, **kwargs)
        return {name + '_rpn': loss for name, loss in losses.items()}


@HEADS.register_module()
class FCOSMultiHead(MultiHeadMixin, FCOSHead):
    pass


@HEADS.register_module()
class FCOSRPNHead(MultiHeadMixin, FCOSHead):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.proposal_cfg = dict(
    #         nms_pre=2000,
    #         max_per_img=1000,
    #         nms=dict(type='nms', iou_threshold=0.7),
    #         min_bbox_size=0,
    #     )

    def loss(self, *args, **kwargs):
        losses = super().loss(*args, **kwargs)
        return {name + '_rpn': loss for name, loss in losses.items()}

    def get_bboxes(self, cls_scores, *args, **kwargs):
        if self.training:
            cls_scores = [
                einops.reduce(
                    cls_score,
                    'bs num_classes h w -> bs 1 h w',
                    reduction='max',
                    num_classes=self.num_classes,
                ) for cls_score in cls_scores
            ]
            with WithPos(self.prior_generator):
                return super().get_bboxes(cls_scores, *args, **kwargs)
        return super().get_bboxes(cls_scores, *args, **kwargs)

    def _get_bboxes_single(self, *args, **kwargs):
        if self.training:
            return RPNHead._get_bboxes_single(self, *args, **kwargs)
        return super()._get_bboxes_single(*args, **kwargs)

    def _bbox_post_process(self, *args, **kwargs):
        if self.training:
            return RPNHead._bbox_post_process(self, *args, **kwargs)
        return super()._bbox_post_process(*args, **kwargs)


@HEADS.register_module()
class RetinaRPNHead(RetinaMultiHead):

    def get_bboxes(self, cls_scores, *args, **kwargs):
        if self.training:
            cls_scores = [
                einops.reduce(
                    cls_score,
                    'bs (num_anchors num_classes) h w -> bs num_anchors h w',
                    reduction='max',
                    num_classes=self.num_classes,
                ) for cls_score in cls_scores
            ]
            with WithPos(self.prior_generator):
                return super().get_bboxes(cls_scores, *args, **kwargs)
        return super().get_bboxes(cls_scores, *args, **kwargs)

    def _get_bboxes_single(self, *args, **kwargs):
        if self.training:
            return RPNHead._get_bboxes_single(self, *args, **kwargs)
        return super()._get_bboxes_single(*args, **kwargs)

    def _bbox_post_process(self, *args, **kwargs):
        if self.training:
            return RPNHead._bbox_post_process(self, *args, **kwargs)
        return super()._bbox_post_process(*args, **kwargs)
