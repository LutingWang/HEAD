from typing import Optional, Type, TypeVar, cast, overload

import torch
from mmdet.core.bbox import (
    AssignResult,
    BaseSampler,
    SamplingResult,
    RandomSampler as _RandomSampler,
)
from mmdet.core.bbox.builder import BBOX_SAMPLERS

T = TypeVar('T')


class SamplingResultWithBBoxIDs(SamplingResult):
    _pos_bbox_ids: torch.Tensor
    _neg_bbox_ids: torch.Tensor

    @classmethod
    def cast(
        cls: Type[T], sampling_result: SamplingResult, bbox_ids: torch.Tensor,
    ) -> T:
        sampling_result.__class__ = cls
        sampling_result = cast(SamplingResultWithBBoxIDs, sampling_result)
        sampling_result._pos_bbox_ids = bbox_ids[sampling_result.pos_inds]
        sampling_result._neg_bbox_ids = bbox_ids[sampling_result.neg_inds]
        return sampling_result

    @property
    def bbox_ids(self) -> torch.Tensor:
        return torch.cat([self._pos_bbox_ids, self._neg_bbox_ids])


class BBoxIDsMixin(BaseSampler):

    @overload
    def sample(
        self,
        assign_result: AssignResult,
        bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SamplingResult:
        ...

    @overload
    def sample(
        self,
        assign_result: AssignResult,
        bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: Optional[torch.Tensor],
        bbox_ids: torch.Tensor,
        **kwargs,
    ) -> SamplingResultWithBBoxIDs:
        ...

    def sample(
        self,
        assign_result: AssignResult,
        bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: Optional[torch.Tensor] = None,
        bbox_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SamplingResult:
        sampling_result: SamplingResult = super().sample(
            assign_result,
            bboxes,
            gt_bboxes,
            gt_labels,
            **kwargs,
        )
        if bbox_ids is None:
            return sampling_result
        num_gts = sampling_result.num_gts
        gt_bbox_ids = bbox_ids.new_full(
            (num_gts, bbox_ids.shape[1]), -1,
        )
        torch.arange(num_gts, out=gt_bbox_ids[:, -1])
        bbox_ids = torch.cat([gt_bbox_ids, bbox_ids], dim=0)
        return SamplingResultWithBBoxIDs.cast(
            sampling_result, bbox_ids,
        )


@BBOX_SAMPLERS.register_module(force=True)
class RandomSampler(BBoxIDsMixin, _RandomSampler):
    pass
