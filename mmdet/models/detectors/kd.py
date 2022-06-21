# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Tuple

import einops
import mmcv
import todd
# import todd.adapts
# import todd.distillers
# import todd.schedulers
# import todd.utils
import torch
import torch.nn.functional as F

from ...core import (MaxIoUAssigner, RandomSampler, SamplingResult,
                     bbox2result, bbox2roi)
from ..builder import DETECTORS, build_head, build_neck
from ..dense_heads import RetinaHead, RetinaRPNHead
from ..roi_heads import StandardRoIHead
from .single_stage import SingleStageDetector
from .two_stage import TwoStageDetector


@todd.adapts.ADAPTS.register_module()
class CustomAdapt(todd.adapts.BaseAdapt):

    def __init__(self, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        self._stride = stride

    def forward(
        self,
        feat: torch.Tensor,
        pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_idx = pos[:, 1] >= 0
        feat = feat[valid_idx]
        pos = pos[valid_idx]
        bs, level, h, w, id_ = pos.split(1, 1)
        h = h // self._stride
        w = w // self._stride
        pos = torch.cat((level, bs, h, w), dim=-1)
        id_ = id_.reshape(-1)
        return feat, pos, id_


def sample(
    roi_head: StandardRoIHead,
    x: List[torch.Tensor],
    img_metas: List[dict],
    gt_bboxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_bboxes_ignore: Optional[torch.Tensor],
    proposal_list: List[torch.Tensor],
):
    bbox_assigner: MaxIoUAssigner = roi_head.bbox_assigner
    bbox_sampler: RandomSampler = roi_head.bbox_sampler
    num_imgs = len(img_metas)
    if gt_bboxes_ignore is None:
        gt_bboxes_ignore = [None for _ in range(num_imgs)]
    sampling_results: List[SamplingResult] = []
    bbox_ids = []
    for i in range(num_imgs):
        bboxes = proposal_list[i][:, :4]
        ids = proposal_list[i][:, 5:]
        assign_result = bbox_assigner.assign(
            bboxes,
            gt_bboxes[i],
            gt_bboxes_ignore[i],
            gt_labels[i],
        )
        sampling_result: SamplingResult = bbox_sampler.sample(
            assign_result,
            bboxes,
            gt_bboxes[i],
            gt_labels[i],
            ids,
            feats=[lvl_feat[i][None] for lvl_feat in x],
        )
        ids = sampling_result.bbox_ids
        ids = torch.cat([ids.new_full((ids.shape[0], 1), i), ids], dim=-1)
        sampling_results.append(sampling_result)
        bbox_ids.append(ids)
    bboxes = [res.bboxes for res in sampling_results]
    bbox_ids = torch.cat(bbox_ids)
    return bboxes, bbox_ids, sampling_results


class MultiHeadMixin:

    def __init__(
        self,
        *args,
        visualize: bool = False,
        warmup: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._visualize = visualize
        self._warmup = None if warmup is None else {
            name: todd.schedulers.SCHEDULERS.build(cfg)
            for name, cfg in warmup.items()
        }

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        *args,
        **kwargs,
    ) -> Tuple[Dict[str, Any], List[SamplingResult]]:
        if self._visualize:
            imgs = einops.rearrange(img, 'bs c h w -> bs h w c')
            imgs = imgs.detach().cpu().numpy()
            self.imgs = [
                mmcv.imdenormalize(
                    img,
                    mean=img_meta['img_norm_cfg']['mean'],
                    std=img_meta['img_norm_cfg']['std'],
                    to_bgr=img_meta['img_norm_cfg']['to_rgb'],
                ) for img, img_meta in zip(imgs, img_metas)
            ]
        outs = super().forward_train(img, img_metas, *args, **kwargs)
        losses = outs[0] if isinstance(outs, tuple) else outs
        if self._warmup is not None:
            losses.update({
                name: losses[name] * scheduler
                for name, scheduler in self._warmup.items()
            })
        return outs


@DETECTORS.register_module()
class MultiHeadSingleStageDetector(MultiHeadMixin, SingleStageDetector):
    pass


@DETECTORS.register_module()
class MultiHeadTwoStageDetector(MultiHeadMixin, TwoStageDetector):
    roi_head: StandardRoIHead
    rpn_head: RetinaRPNHead
    train_cfg: dict
    test_cfg: dict

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
        proposals: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], List[SamplingResult]]:
        roi_head: StandardRoIHead = self.roi_head
        x = self.extract_feat(img)
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg,
            **kwargs,
        )

        self.bboxes, self.bbox_ids, sampling_results = sample(roi_head, x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, proposal_list)

        bbox_results = self.roi_head._bbox_forward_train(
            x,
            sampling_results,
            gt_bboxes,
            gt_labels,
            img_metas,
        )
        roi_losses = bbox_results['loss_bbox']

        losses = {**rpn_losses, **roi_losses}
        return losses

    def simple_test(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        rescale: bool = False,
    ) -> list:
        feat = self.extract_feat(img)
        results_list = self.rpn_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.rpn_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(
        self,
        imgs: List[torch.Tensor],
        img_metas: List[dict],
        rescale: bool = False,
    ) -> list:
        feats = self.extract_feats(imgs)
        results_list = self.rpn_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.rpn_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results


@DETECTORS.register_module()
@todd.distillers.SelfDistiller.wrap()
class HEAD(MultiHeadTwoStageDetector):
    distiller: todd.distillers.SelfDistiller

    def forward_train(self, *args, **kwargs) -> Dict[str, Any]:
        losses = super().forward_train(*args, **kwargs)
        kd_losses = self.distiller.distill()
        return {**losses, **kd_losses}

    def forward_test(self, *args, **kwargs) -> Any:
        results = super().forward_test(*args, **kwargs)
        self.distiller.reset()
        return results


class SingleTeacherDistiller(todd.distillers.SingleTeacherDistiller):

    def __init__(self, *args, teacher: dict, **kwargs):
        teacher_model = todd.utils.ModelLoader.load_mmlab_models(DETECTORS, **teacher)
        super().__init__(*args, teacher=teacher_model, **kwargs)


@DETECTORS.register_module()
@todd.distillers.SingleTeacherDistiller.wrap()
class GID(SingleStageDetector):
    distiller: todd.distillers.SelfDistiller

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        losses = super().forward_train(img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        kd_losses = self.distiller.distill(dict())
        return {**losses, **kd_losses}

    def forward_test(self, *args, **kwargs) -> Any:
        results = super().forward_test(*args, **kwargs)
        self.distiller.reset()
        return results


@DETECTORS.register_module()
@todd.distillers.SelfDistiller.wrap()
class LabelEnc(SingleStageDetector):
    distiller: todd.distillers.SelfDistiller
    bbox_head: RetinaHead

    def __init__(self, *args, neck: dict = None, **kwargs):
        super().__init__(*args, neck=neck, **kwargs)
        self._masks = todd.adapts.LabelEncMask(num_classes=80, aug=True)
        self._backbone = todd.adapts.LabelEncAdapt(base_channels=8)
        self._neck = build_neck(neck)

    def extract_label_feat(
        self,
        img: torch.Tensor,
        bboxes: List[torch.Tensor],
        labels: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        masks = self._masks(
            shape=tuple(img[0].shape[-2:]),
            bboxes=bboxes, labels=labels,
        )
        x = self._backbone(masks)
        x = self._neck(x)
        return x

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        losses = super().forward_train(img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        label_neck = self.extract_label_feat(img, gt_bboxes, gt_labels)
        label_losses = self.bbox_head.forward_train(label_neck, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        label_losses = {k + '_label': v for k, v in label_losses.items()}
        kd_losses = self.distiller.distill(dict(
            label_neck=label_neck,
        ))
        return {**losses, **label_losses, **kd_losses}

    def forward_test(self, *args, **kwargs) -> Any:
        results = super().forward_test(*args, **kwargs)
        self.distiller.reset()
        return results


@DETECTORS.register_module()
@SingleTeacherDistiller.wrap()
class GDetSingleStage(SingleStageDetector):
    distiller: SingleTeacherDistiller
    bbox_head: RetinaRPNHead

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proposal_cfg = mmcv.ConfigDict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        )
        self.roi_head: StandardRoIHead = build_head(mmcv.ConfigDict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[8, 16, 32, 64],
            ),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            ),
            train_cfg=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False,  # NOTE
                ),
                pos_weight=-1,
                debug=False,
            ),
            test_cfg=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100,
            ),
        ))
        todd.utils.freeze_model(self.roi_head)

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        teacher_model: TwoStageDetector = self.distiller.teacher

        x = self.extract_feat(img)
        losses, proposal_list = self.bbox_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=self._proposal_cfg,
        )
        with torch.no_grad():
            teacher_x = teacher_model.extract_feat(img)

        bboxes, bbox_ids, _ = sample(self.roi_head, teacher_x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, proposal_list)

        rois = bbox2roi(bboxes)
        _ = self.roi_head.bbox_roi_extractor(
            teacher_x[:self.roi_head.bbox_roi_extractor.num_inputs], rois,
        )
        kd_losses = self.distiller.distill(
            dict(
                bbox_ids=bbox_ids,
                bboxes=bboxes,
            ))
        return {**losses, **kd_losses}

    def forward_test(self, *args, **kwargs) -> Any:
        results = super().forward_test(*args, **kwargs)
        self.distiller.reset()
        return results


@DETECTORS.register_module()
@SingleTeacherDistiller.wrap()
class GDet(SingleStageDetector):
    distiller: SingleTeacherDistiller
    bbox_head: RetinaRPNHead

    def __init__(self, *args, with_ckd: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._with_ckd = with_ckd
        self._proposal_cfg = mmcv.ConfigDict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        )

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        teacher_model: TwoStageDetector = self.distiller.teacher
        roi_head: StandardRoIHead = teacher_model.roi_head

        x = self.extract_feat(img)
        losses, proposal_list = self.bbox_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=self._proposal_cfg,
        )
        with torch.no_grad():
            teacher_x = teacher_model.extract_feat(img)

        bboxes, bbox_ids, _ = sample(roi_head, teacher_x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, proposal_list)

        rois = bbox2roi(bboxes)
        if self._with_ckd:
            with torch.no_grad():
                _ = roi_head._bbox_forward(teacher_x, rois)
        else:
            _ = roi_head.bbox_roi_extractor(
                teacher_x[:roi_head.bbox_roi_extractor.num_inputs], rois,
            )
        kd_losses = self.distiller.distill(
            dict(
                bbox_ids=bbox_ids,
                bboxes=bboxes,
            ))
        return {**losses, **kd_losses}

    def forward_test(self, *args, **kwargs) -> Any:
        results = super().forward_test(*args, **kwargs)
        self.distiller.reset()
        return results


@DETECTORS.register_module()
@SingleTeacherDistiller.wrap()
class TB_HEADSingleStage(MultiHeadSingleStageDetector):
    distiller: SingleTeacherDistiller

    def forward_train(self, img: torch.Tensor, *args, **kwargs) -> Tuple[Dict[str, Any], List[SamplingResult]]:
        losses = super().forward_train(img, *args, **kwargs)
        with torch.no_grad():
            teacher_model: SingleStageDetector = self.distiller.teacher
            teacher_x = teacher_model.extract_feat(img)
            _ = teacher_model.bbox_head(teacher_x)
        kd_losses = self.distiller.distill()
        return {**losses, **kd_losses}

    def forward_test(self, *args, **kwargs) -> Any:
        results = super().forward_test(*args, **kwargs)
        self.distiller.reset()
        return results


@DETECTORS.register_module()
@SingleTeacherDistiller.wrap()
class TB_HEAD(MultiHeadTwoStageDetector):
    distiller: SingleTeacherDistiller

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: torch.Tensor,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        losses = super().forward_train(
            img,
            img_metas,
            gt_bboxes,
            *args,
            **kwargs)
        with torch.no_grad():
            teacher_model: TwoStageDetector = self.distiller.teacher
            teacher_x = teacher_model.extract_feat(img)
            rois = bbox2roi(self.bboxes)
            teacher_model.roi_head._bbox_forward(teacher_x, rois)

        if self._visualize:
            self.distiller.visualize()
        kd_losses = self.distiller.distill(
            dict(
                bboxes=self.bboxes,
                bbox_ids=self.bbox_ids,
                gt_bboxes=gt_bboxes,
                batch_input_shape=tuple(img[0].shape[-2:])))

        return {**losses, **kd_losses}

    def forward_test(self, *args, **kwargs) -> Any:
        results = super().forward_test(*args, **kwargs)
        self.distiller.reset()
        return results


@DETECTORS.register_module()
@todd.distillers.SelfDistiller.wrap()
class SelfSingleStageDetector(MultiHeadSingleStageDetector):
    distiller: todd.distillers.SelfDistiller

    def forward_train(self, *args, **kwargs) -> Dict[str, Any]:
        losses = super().forward_train(*args, **kwargs)
        kd_losses = self.distiller.distill()
        return {**losses, **kd_losses}

    def forward_test(self, *args, **kwargs) -> Any:
        results = super().forward_test(*args, **kwargs)
        self.distiller.reset()
        return results


@DETECTORS.register_module()
@SingleTeacherDistiller.wrap()
class TB_HEADTwoStage(TwoStageDetector):
    distiller: SingleTeacherDistiller

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cascade_roi_head = build_head(mmcv.ConfigDict(
            type='CascadeRoIHead',
            num_stages=3,
            stage_loss_weights=[1, 0.5, 0.25],
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=[
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=80,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                   loss_weight=1.0)),
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=80,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.05, 0.05, 0.1, 0.1]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                   loss_weight=1.0)),
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=80,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.033, 0.033, 0.067, 0.067]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
            ],
            train_cfg=[
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    pos_weight=-1,
                    debug=False),
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.6,
                        neg_iou_thr=0.6,
                        min_pos_iou=0.6,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    pos_weight=-1,
                    debug=False),
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.7,
                        min_pos_iou=0.7,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    pos_weight=-1,
                    debug=False)
            ],
            test_cfg=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100,
            ),
        ))
        old_func = self.cascade_roi_head._bbox_forward
        def wrapped(stage, x, rois):
            with torch.no_grad():
                teacher_model: TwoStageDetector = self.distiller.teacher
                _ = teacher_model.roi_head._bbox_forward(stage, self.teacher_x, rois)
            return old_func(stage, x, rois)
        self.cascade_roi_head._bbox_forward = wrapped
        self._warmup = todd.schedulers.SchedulerModuleList(dict(
            warmup=dict(
                type='WarmupScheduler',
                tensor_names=[
                    's0.loss_cls', 's0.loss_bbox',
                    's1.loss_cls', 's1.loss_bbox',
                    's2.loss_cls', 's2.loss_bbox',
                ],
                iter_=2000,
                value=1.0,
            ),
        ))

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs):
        with torch.no_grad():
            teacher_model: TwoStageDetector = self.distiller.teacher
            self.teacher_x = teacher_model.extract_feat(img)
        x = self.extract_feat(img)
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                          self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg,
            **kwargs)

        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs,
        )
        # bboxes, bbox_ids, sampling_results = sample(self.roi_head, x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, proposal_list)
        # bbox_results = self.roi_head._bbox_forward_train(x, sampling_results, gt_bboxes, gt_labels, img_metas)
        cascade_roi_losses = self.cascade_roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs,
        )
        cascade_roi_losses = self._warmup(cascade_roi_losses)
        cascade_roi_losses = {k + '_roi': v for k, v in cascade_roi_losses.items()}

        kd_losses = self.distiller.distill(dict(
            # bboxes=bboxes, bbox_ids=bbox_ids,
        ))
        return {**rpn_losses, **roi_losses, **cascade_roi_losses, **kd_losses}


@DETECTORS.register_module()
@SingleTeacherDistiller.wrap()
class GDetFasterRCNN(TwoStageDetector):
    distiller: SingleTeacherDistiller

    def __init__(self, *args, with_ckd: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_ckd = with_ckd

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs):
        x = self.extract_feat(img)

        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg,
            **kwargs)

        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = self.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)
        bboxes = [res.bboxes for res in sampling_results]

        bbox_results = self.roi_head._bbox_forward_train(x, sampling_results,
                                                gt_bboxes, gt_labels,
                                                img_metas)

        with torch.no_grad():
            teacher_model: TwoStageDetector = self.distiller.teacher
            teacher_x = teacher_model.extract_feat(img)
            rois = bbox2roi(bboxes)
            if self.with_ckd:
                _ = teacher_model.roi_head._bbox_forward(0, teacher_x, rois)
            else:
                _ = teacher_model.roi_head.bbox_roi_extractor[0](
                    teacher_x[:teacher_model.roi_head.bbox_roi_extractor[0].num_inputs], rois,
                )
        kd_losses = self.distiller.distill(dict(bboxes=bboxes))

        return {**rpn_losses, **bbox_results['loss_bbox'], **kd_losses}

    def forward_test(self, *args, **kwargs) -> Any:
        results = super().forward_test(*args, **kwargs)
        self.distiller.reset()
        return results


@DETECTORS.register_module()
@SingleTeacherDistiller.wrap()
class SingleTeacherSingleStageDetector(MultiHeadSingleStageDetector):
    distiller: SingleTeacherDistiller
    bbox_head: RetinaHead

    def __init__(self, *args, dev1: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._dev1 = dev1
        if dev1:
            # from ..backbones.resnext import Bottleneck
            # self._block = Bottleneck(
            #     256, 64, stride=2,
            #     downsample=nn.AvgPool2d(
            #         kernel_size=2, stride=2,
            #         ceil_mode=True, count_include_pad=False),
            #     dcn=dict(
            #         type='DCN', deform_groups=1, fallback_on_stride=False),
            #     init_cfg=[
            #         dict(type='Kaiming', layer='Conv2d'),
            #         dict(
            #             type='Constant', val=1,
            #             layer=['_BatchNorm', 'GroupNorm'],
            #         ),
            #     ],
            # )
            from todd.schedulers import StepScheduler
            # self._block = nn.AvgPool2d(
            #     kernel_size=2,
            #     stride=2,
            #     ceil_mode=True,
            #     count_include_pad=False)
            self._scheduler = StepScheduler(value=0.05, iters=[7330 * 4])
            self._teacher_online = True

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], List[SamplingResult]]:
        teacher_model: TwoStageDetector = self.distiller.teacher
        # if get_iter() >= 7330 * 6:
        if self._dev1 and self._teacher_online and todd.utils.get_iter() >= 7330:
            self._teacher_online = False
            todd.utils.freeze_model(teacher_model)
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        if self._dev1 and self._teacher_online:
            teacher_x = teacher_model.extract_feat(img)
        else:
            with torch.no_grad():
                teacher_x = teacher_model.extract_feat(img)
                teacher_model.rpn_head(teacher_x)

        custom_tensors = dict(
            gt_bboxes=gt_bboxes,
            batch_input_shape=img_metas[0]['batch_input_shape'],
        )

        if self._dev1:
            if self._teacher_online:
                # teacher_x = [self._block(f) for f in teacher_x]
                teacher_x = [
                    F.adaptive_avg_pool2d(teacher_f, f.shape[-2:])
                    for teacher_f, f in zip(teacher_x, x)
                ]
                lr = self._scheduler.value
                teacher_x = [f * lr + f.detach() * (1 - lr) for f in teacher_x]
                custom_tensors['teacher_neck'] = [
                    f.detach() for f in teacher_x
                ]
            else:
                # teacher_x = [self._block(f) for f in teacher_x]
                teacher_x = [
                    F.adaptive_avg_pool2d(teacher_f, f.shape[-2:])
                    for teacher_f, f in zip(teacher_x, x)
                ]
                teacher_x = [f.detach() for f in teacher_x]
                custom_tensors['teacher_neck'] = teacher_x

        kd_losses = self.distiller.distill(custom_tensors)
        losses.update(kd_losses)
        if self._dev1 and self._teacher_online:
            aux_losses = self.bbox_head.forward_train(
                teacher_x,
                img_metas,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
            )
            aux_losses = {
                name + '_aux': loss
                for name, loss in aux_losses.items()
            }
            losses.update(aux_losses)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        # if self._dev1:
        #     teacher_model: TwoStageDetector = self.distiller.teacher
        #     teacher_x = teacher_model.extract_feat(img)
        #     feat = [self._block(f) for f in teacher_x]
        #     results_list = self.bbox_head.simple_test(
        #         feat, img_metas, rescale=rescale)
        #     bbox_results = [
        #         bbox2result(
        #             det_bboxes, det_labels, self.bbox_head.num_classes)
        #         for det_bboxes, det_labels in results_list
        #     ]
        #     return bbox_results
        return super().simple_test(img, img_metas, rescale)

    def forward_test(self, *args, **kwargs) -> Any:
        results = super().forward_test(*args, **kwargs)
        self.distiller.reset()
        return results


@DETECTORS.register_module()
@SingleTeacherDistiller.wrap()
class SingleTeacherTwoStageDetector(TwoStageDetector):
    distiller: SingleTeacherDistiller
    bbox_head: RetinaHead

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: torch.Tensor,
        *args, **kwargs,
    ) -> Tuple[Dict[str, Any], List[SamplingResult]]:
        teacher_model: TwoStageDetector = self.distiller.teacher
        losses = super().forward_train(img, img_metas, gt_bboxes, *args, **kwargs)
        with torch.no_grad():
            teacher_x = teacher_model.extract_feat(img)

        custom_tensors = dict(
            gt_bboxes=gt_bboxes,
            batch_input_shape=tuple(img[0].shape[-2:]),
        )

        kd_losses = self.distiller.distill(custom_tensors)
        losses.update(kd_losses)
        return losses

    def forward_test(self, *args, **kwargs) -> Any:
        results = super().forward_test(*args, **kwargs)
        self.distiller.reset()
        return results
