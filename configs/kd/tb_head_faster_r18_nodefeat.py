_base_ = [
    'retina_faster_r18.py',
]

model = dict(
    type='TB_HEAD',
    warmup=dict(
        loss_cls=dict(
            type='WarmupScheduler',
            iter_=2000,
        ),
        loss_bbox=dict(
            type='WarmupScheduler',
            iter_=2000,
        ),
    ),
    distiller=dict(
        teacher=dict(
            config='configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
            ckpt='data/ckpts/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth',
        ),
        weight_transfer={
            'student.roi_head': 'teacher.roi_head',
        },
        student_hooks=dict(
            retina_cls=dict(
                type='MultiCallsHook',
                path='rpn_head.cls_convs[-1]',
            ),
            rcnn_bbox=dict(
                type='StandardHook',
                path='roi_head.bbox_head.shared_fcs[-1]',
            ),
            rcnn_bbox_aux=dict(
                type='StandardHook',
                path='roi_head.bbox_head.shared_fcs[0]',
            ),
        ),
        teacher_hooks=dict(
            teacher_rcnn_bbox_aux=dict(
                type='StandardHook',
                path='roi_head.bbox_head.shared_fcs[0]',
            ),
            teacher_rcnn_bbox=dict(
                type='StandardHook',
                path='roi_head.bbox_head.shared_fcs[-1]',
            ),
        ),
        adapts={
            'retina_cls_reshaped':
            dict(
                type='Rearrange',
                fields=['retina_cls'],
                parallel=True,
                pattern='bs dim h w -> bs h w dim',
            ),
            ('teacher_rcnn_bbox_filtered', 'bbox_poses', 'anchor_ids'):
            dict(
                type='CustomAdapt',
                fields=['teacher_rcnn_bbox', 'bbox_ids'],
                stride=1,
            ),
            'retina_cls_indexed':
            dict(
                type='Index',
                fields=['retina_cls_reshaped', 'bbox_poses'],
            ),
            'retina_cls_decoupled':
            dict(
                type='Decouple',
                fields=['retina_cls_indexed', 'anchor_ids'],
                num=9,
                in_features=256,
                out_features=1024,
            ),
            'rcnn_bbox_aux_adapted':
            dict(
                type='Linear',
                fields=['rcnn_bbox_aux'],
                in_features=1024,
                out_features=1024,
            ),
            'rcnn_bbox_adapted':
            dict(
                type='Linear',
                fields=['rcnn_bbox'],
                in_features=1024,
                out_features=1024,
            ),
        },
        losses=dict(
            loss_cross=dict(
                type='MSELoss',
                fields=['retina_cls_decoupled', 'teacher_rcnn_bbox_filtered'],
                weight=2.0,
            ),
            loss_mimic_rcnn_aux=dict(
                type='MSELoss',
                fields=['rcnn_bbox_aux_adapted', 'teacher_rcnn_bbox_aux'],
                weight=dict(
                    type='WarmupScheduler',
                    value=2.0,
                    iter_=2000,
                ),
            ),
            loss_mimic_rcnn=dict(
                type='MSELoss',
                fields=['rcnn_bbox_adapted', 'teacher_rcnn_bbox'],
                weight=dict(
                    type='WarmupScheduler',
                    value=2.0,
                    iter_=2000,
                ),
            ),
        ),
    ),
)
