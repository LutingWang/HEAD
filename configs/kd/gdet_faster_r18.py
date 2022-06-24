_base_ = [
    'faster_rcnn_r18_fpn_mstrain_1x_coco.py',
]

model = dict(
    type='GDetFasterRCNN',
    distiller=dict(
        teacher=dict(
            config='configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py',
            ckpt='data/ckpts/cascade_rcnn_r50_fpn_mstrain_3x_coco.pth',
        ),
        student_hooks=dict(
            neck=dict(type='StandardHook', path='neck'),
            rcnn_bbox=dict(type='StandardHook', path='roi_head.bbox_head.shared_fcs[-1]')),
        teacher_hooks=dict(
            teacher_roi_feats=dict(
                type='StandardHook',
                path='roi_head.bbox_roi_extractor[0]',
            ),
            teacher_rcnn_bbox=dict(
                type='StandardHook',
                path='roi_head.bbox_head[0].shared_fcs[-1]',
            ),
        ),
        adapts={
            'rcnn_bbox_adapted':
            dict(
                type='Linear',
                fields=['rcnn_bbox'],
                in_features=1024,
                out_features=1024,
                bias=False,
            ),
            'roi_feats':
            dict(
                type='RoIAlign',
                fields=['neck', 'bboxes'],
                strides=[4, 8, 16, 32, 64],
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
            loss_ckd=dict(
                type='CKDLoss',
                fields=[
                    'rcnn_bbox_adapted', 'teacher_rcnn_bbox', 'bboxes',
                ],
                weight=0.5,
            ),
            loss_sgfi=dict(
                type='SGFILoss',
                fields=['roi_feats_adapted', 'teacher_roi_feats'],
                weight=1.0,
            ),
        ),
    ),
)
