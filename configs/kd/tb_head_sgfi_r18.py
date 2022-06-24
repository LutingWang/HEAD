_base_ = [
    'head_r18.py',
]

model = dict(
    type='TB_HEAD',
    distiller=dict(
        teacher=dict(
            config='configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
            ckpt='data/ckpts/faster_rcnn_r50_fpn_mstrain_2x_coco_cos.pth',
        ),
        student_hooks=dict(neck=dict(type='StandardHook', path='neck')),
        student_trackings=dict(bboxes=dict(type='StandardHook', path='bboxes')),
        teacher_hooks=dict(
            teacher_roi_feats=dict(
                type='StandardHook',
                path='roi_head.bbox_roi_extractor',
            ),
            teacher_rcnn_bbox=dict(
                type='StandardHook',
                path='roi_head.bbox_head.shared_fcs[-1]',
            ),
        ),
        adapts={
            'rcnn_bbox_adapted':
            dict(
                type='Linear',
                fields=['rcnn_bbox'],
                in_features=1024,
                out_features=1024,
            ),
            'roi_feats':
            dict(
                type='RoIAlign',
                fields=['neck', 'bboxes'],
                strides=[8, 16, 32, 64, 128],
            ),
            'roi_feats_adapted':
            dict(
                type='Conv2d',
                fields=['roi_feats'],
                parallel=True,
                in_channels=256,
                out_channels=256,
                kernel_size=1,
            ),
        },
        losses=dict(
            loss_mimic_sgfi=dict(
                type='SGFILoss',
                fields=['roi_feats_adapted', 'teacher_roi_feats'],
                weight=1.0,
            ),
            loss_mimic_rcnn=dict(
                type='MSELoss',
                fields=['rcnn_bbox_adapted', 'teacher_rcnn_bbox'],
                weight=2.0,
            ),
        ),
        schedulers=dict(
            warmup=dict(fields=[
                'loss_self_fcos', 'loss_self_rcnn', 'loss_mimic_rcnn',
            ])),
    ),
)
