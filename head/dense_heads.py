from typing import Any, Dict, List, Optional, Tuple, TypeVar, overload

import einops
import todd
import torch
from mmcv.ops import batched_nms
from mmdet.core import BaseBBoxCoder
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads import AnchorFreeHead, AnchorHead
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.retina_head import RetinaHead
from mmdet.models.dense_heads.rpn_head import RPNHead as _RPNHead


class CacheAnchorsMixin(AnchorHead):

    def __init__(
        self,
        *args,
        cache_anchors: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._cache_anchors = cache_anchors

    @property
    def cache_anchors(self) -> bool:
        return getattr(self, '_cache_anchors', False)

    def get_anchors(
        self,
        featmap_sizes: List[Tuple[int, int]],
        *args,
        **kwargs,
    ):
        anchor_list, valid_flag_list = super().get_anchors(
            featmap_sizes, *args, **kwargs,
        )
        if self.cache_anchors:
            anchors: List[torch.Tensor] = anchor_list[0]
            self.anchors = [  # yapf: disable
                anchor.reshape(featmap_size + (3, 4))
                for featmap_size, anchor in zip(featmap_sizes, anchors)
            ]
        return anchor_list, valid_flag_list


@HEADS.register_module(force=True)
class RPNHead(CacheAnchorsMixin, _RPNHead):
    bbox_coder: BaseBBoxCoder

    def _bbox_post_process(
        self,
        mlvl_scores: List[torch.Tensor],
        mlvl_bboxes: List[torch.Tensor],
        mlvl_valid_anchors: List[torch.Tensor],
        level_ids: List[torch.Tensor],
        cfg,
        img_shape: Tuple[int, int],
        **kwargs,
    ):
        if not getattr(self.prior_generator, 'with_pos', False):
            return super()._bbox_post_process(
                mlvl_scores,
                mlvl_bboxes,
                mlvl_valid_anchors,
                level_ids,
                cfg,
                img_shape,
                **kwargs,
            )
        scores = torch.cat(mlvl_scores)
        bboxes = torch.cat(mlvl_bboxes)
        anchors = torch.cat(mlvl_valid_anchors)
        ids = torch.cat(level_ids)

        poses = anchors[:, -4:]
        anchors = anchors[:, :-4]
        proposals: torch.Tensor = self.bbox_coder.decode(
            anchors,
            bboxes,
            max_shape=img_shape,
        )

        if cfg.min_bbox_size >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = torch.logical_and(
                w > cfg.min_bbox_size,
                h > cfg.min_bbox_size,
            )
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]
                poses = poses[valid_mask]

        if proposals.numel() == 0:
            return proposals.new_zeros(0, 9)
        dets, keep = batched_nms(proposals, scores, ids, cfg.nms)
        dets = dets[:cfg.max_per_img]
        keep = keep[:cfg.max_per_img]
        poses = poses[keep]
        return torch.cat([dets, poses], dim=-1)


T = TypeVar('T', AnchorHead, AnchorFreeHead)


class RPNMixin(BaseDenseHead):

    @overload
    def forward_train(
        self,
        x: torch.Tensor,
        img_metas: List[Dict[str, Any]],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        pass

    @overload
    def forward_train(
        self,
        x: torch.Tensor,
        img_metas: List[Dict[str, Any]],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor],
        proposal_cfg: todd.base.Config,
    ) -> Tuple[Dict[str, Any], List[Any]]:
        pass

    def forward_train(
        self: T,
        x: torch.Tensor,
        img_metas: List[Dict[str, Any]],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor] = None,
        proposal_cfg: Optional[todd.base.Config] = None,
    ):
        if proposal_cfg is None:
            return super().forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
            )
        cls_scores, *outs = self(x)
        losses: Dict[str, Any] = self.loss(
            cls_scores,
            *outs,
            gt_bboxes,
            gt_labels,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore,
        )
        losses = {f'{k}_rpn': v for k, v in losses.items()}
        cls_scores = [
            einops.reduce(
                cls_score,
                'bs (num_anchors num_classes) h w -> bs num_anchors h w',
                reduction='max',
                num_classes=self.num_classes,
            ) for cls_score in cls_scores
        ]
        with \
                todd.setattr_temp(self.prior_generator, 'with_pos', True), \
                todd.setattr_temp(self, '__class__', RPNHead):
            proposal_list = self.get_bboxes(
                cls_scores,
                *outs,
                img_metas=img_metas,
                cfg=proposal_cfg,
            )
        return losses, proposal_list


@HEADS.register_module()
class RetinaRPNHead(RPNMixin, RetinaHead):
    pass
