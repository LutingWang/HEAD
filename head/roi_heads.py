from typing import Any, Dict, List, Optional

import torch
from mmdet.core import BaseAssigner
from mmdet.models import StandardRoIHead
from mmdet.models.builder import HEADS

from .samplers import BBoxIDsMixin, SamplingResultWithBBoxIDs


@HEADS.register_module()
class StandardRoIHeadWithBBoxIDs(StandardRoIHead):
    bbox_assigner: BaseAssigner
    bbox_sampler: BBoxIDsMixin

    def _sample(
        self,
        x: torch.Tensor,
        proposal_list: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes_ignore: List[Optional[torch.Tensor]],
    ):
        sampling_results: List[SamplingResultWithBBoxIDs] = []
        for i, (proposal, gt_bbox, gt_label, gt_bbox_ignore) in enumerate(
            zip(proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore),
        ):
            feats = [
                lvl_feat[i][None] for lvl_feat in x
            ]  # TODO: is this pythonic?
            proposal_id = proposal[:, 5:]
            proposal = proposal[:, :4]
            assign_result = self.bbox_assigner.assign(
                proposal,
                gt_bbox,
                gt_bbox_ignore,
                gt_label,
            )
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal,
                gt_bbox,
                gt_label,
                proposal_id,
                feats=feats,
            )
            sampling_results.append(sampling_result)

        bboxes = [s.bboxes for s in sampling_results]
        bboxes_ids = torch.cat([s.bbox_ids for s in sampling_results])
        bboxes_img_id = torch.cat([  # yapf: disable
            torch.full_like(s.bbox_ids[:, [0]], i)
            for i, s in enumerate(sampling_results)
        ])
        bboxes_ids = torch.cat((bboxes_img_id, bboxes_ids), dim=-1)

        self.bboxes = bboxes
        self.bboxes_ids = bboxes_ids
        return sampling_results

    def init_weights(self) -> None:
        # It is fine to have this function called after initialization, since
        # `todd.distiller` may transfer weights.
        if not self.is_init:
            super().init_weights()

    def forward_train(
        self,
        x: torch.Tensor,
        img_metas: List[Dict[str, Any]],
        proposal_list: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
        gt_masks: Optional[List[Any]] = None,
        **kwargs,
    ):
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(len(img_metas))]

        losses = dict()

        if self.with_bbox or self.with_mask:
            sampling_results = self._sample(
                x,
                proposal_list,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
            )

        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                x,
                sampling_results,
                gt_bboxes,
                gt_labels,
                img_metas,
            )
            losses.update(bbox_results['loss_bbox'])

        if self.with_mask:
            mask_results = self._mask_forward_train(
                x,
                sampling_results,
                bbox_results['bbox_feats'],
                gt_masks,
                img_metas,
            )
            losses.update(mask_results['loss_mask'])

        return losses
