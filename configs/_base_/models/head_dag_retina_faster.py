_base_ = [
    'retina_faster.py',
]

model = dict(
    type='CrossStageHEAD',
    distiller=dict(
        teacher=dict(
            config='configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
            ckpt=(
                'pretrained/mmdetection/faster_rcnn_r50_fpn_mstrain_3x_coco_'
                '20210524_110822-e10bd31c.pth'
            ),
        ),
        weight_transfer={
            '.student.roi_head': '.teacher.roi_head',
        },
        student_hooks=dict(
            rcnn_bbox=dict(
                type='StandardHook',
                path='.roi_head.bbox_head.shared_fcs[-1]',
            ),
            rcnn_bbox_aux=dict(
                type='StandardHook',
                path='.roi_head.bbox_head.shared_fcs[0]',
            ),
        ),
        teacher_hooks=dict(
            teacher_rcnn_bbox_aux=dict(
                type='StandardHook',
                path='.roi_head.bbox_head.shared_fcs[0]',
            ),
            teacher_rcnn_bbox=dict(
                type='StandardHook',
                path='.roi_head.bbox_head.shared_fcs[-1]',
            ),
        ),
        adapts={
            'rcnn_bbox_aux_adapted': dict(
                type='Linear',
                fields=['rcnn_bbox_aux'],
                in_features=1024,
                out_features=1024,
            ),
            'rcnn_bbox_adapted': dict(
                type='Linear',
                fields=['rcnn_bbox'],
                in_features=1024,
                out_features=1024,
            ),
        },
        losses=dict(
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
