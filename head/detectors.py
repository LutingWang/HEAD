from typing import Any, Dict, List, NoReturn, Optional

import einops
import mmcv
import todd
import torch
from mmdet.core import bbox2result, bbox2roi
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    SingleStageDetector,
    StandardRoIHead,
    TwoStageDetector,
)
from todd.base.iters import inc_iter

from .dense_heads import RPNMixin
from .roi_heads import StandardRoIHeadWithBBoxIDs


class CacheImgsMixin(BaseDetector):

    def __init__(self, *args, cache_imgs: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cache_imgs = cache_imgs

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[Dict[str, Any]],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if self._cache_imgs:
            imgs = einops.rearrange(img, 'bs c h w -> bs h w c')
            imgs = imgs.detach().cpu().numpy()
            self.imgs = [  # yapf: disable
                mmcv.imdenormalize(
                    img,
                    mean=img_meta['img_norm_cfg']['mean'],
                    std=img_meta['img_norm_cfg']['std'],
                    to_bgr=img_meta['img_norm_cfg']['to_rgb'],
                ) for img, img_meta in zip(imgs, img_metas)
            ]
        return super().forward_train(
            img,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks,
            **kwargs,
        )


class SchedulersMixin(BaseDetector):

    def __init__(
        self,
        *args,
        schedulers: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._schedulers = None if schedulers is None else {  # yapf: disable
            name: todd.schedulers.SCHEDULERS.build(cfg)
            for name, cfg in schedulers.items()
        }

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[Dict[str, Any]],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        losses = super().forward_train(
            img,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks,
            **kwargs,
        )
        if self._schedulers is not None:
            losses.update({  # yapf: disable
                name: todd.utils.CollectionTensor.apply(
                    losses[name],
                    lambda loss: loss * scheduler,
                )
                for name, scheduler in self._schedulers.items()
            })
        return losses


@DETECTORS.register_module()
class CrossStageDetector(TwoStageDetector):
    rpn_head: RPNMixin
    roi_head: StandardRoIHead

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[Dict[str, Any]],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        x = self.extract_feat(img)

        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            proposal_cfg,
            **kwargs,
        )

        roi_losses = self.roi_head.forward_train(
            x,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks,
            **kwargs,
        )

        return {**rpn_losses, **roi_losses}

    def simple_test(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        rescale: bool = False,
    ) -> list:
        feat = self.extract_feat(img)
        results_list = self.rpn_head.simple_test(
            feat,
            img_metas,
            rescale=rescale,
        )
        bbox_results = [  # yapf: disable
            bbox2result(det_bboxes, det_labels, self.rpn_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError


class SingleTeacherDistiller(todd.distillers.SingleTeacherDistiller):

    def __init__(self, *args, teacher: todd.base.Config, **kwargs):
        teacher.config = todd.base.Config.load(teacher.config).model
        teacher_model = todd.base.load_open_mmlab_models(
            DETECTORS,
            **teacher,
        )
        super().__init__(*args, teacher=teacher_model, **kwargs)


class DistillerMixin(BaseDetector):
    distiller: SingleTeacherDistiller

    def forward_test(self, *args, **kwargs) -> List[Any]:
        results = super().forward_test(*args, **kwargs)
        self.distiller.reset()
        return results


@DETECTORS.register_module()
@SingleTeacherDistiller.wrap()
class SingleTeacherSingleStageDistiller(SingleStageDetector):

    def forward_train(self, *args, **kwargs) -> Dict[str, Any]:
        with torch.no_grad():
            teacher: TwoStageDetector = self.distiller.teacher
            _ = teacher.forward_train(*args, **kwargs)
        losses = super().forward_train(*args, **kwargs)

        self.distiller.track_tensors()
        kd_losses = self.distiller.distill()
        self.distiller.reset()
        inc_iter()

        # mmdet does not support tuple of losses
        for k, v in kd_losses.items():
            if isinstance(v, tuple):
                kd_losses[k] = list(v)

        return {**losses, **kd_losses}


@DETECTORS.register_module()
@SingleTeacherDistiller.wrap()
class CrossStageHEAD(
    CacheImgsMixin,
    SchedulersMixin,
    DistillerMixin,
    CrossStageDetector,
):
    roi_head: StandardRoIHeadWithBBoxIDs

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[Dict[str, Any]],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        losses = super().forward_train(
            img,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks,
            **kwargs,
        )
        with torch.no_grad():
            teacher: TwoStageDetector = self.distiller.teacher
            teacher_roi_head: StandardRoIHead = teacher.roi_head
            teacher_x = teacher.extract_feat(img)
            rois = bbox2roi(self.roi_head.bboxes)
            teacher_roi_head._bbox_forward(teacher_x, rois)

        custom_tensors = dict(
            bboxes=self.roi_head.bboxes,
            bbox_ids=self.roi_head.bboxes_ids,
            gt_bboxes=gt_bboxes,
            batch_input_shape=tuple(img[0].shape[-2:]),
        )
        self.distiller.track_tensors()
        kd_losses = self.distiller.distill(custom_tensors)
        self.distiller.reset()
        inc_iter()

        return {**losses, **kd_losses}
