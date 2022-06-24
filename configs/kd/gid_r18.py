_base_ = [
    'retinanet_r18_fpn_mstrain_1x_coco.py',
]

model = dict(
    type='SingleTeacherSingleStageDetector',
    bbox_head=dict(
        type='RetinaMultiHead',
        cache_anchors=True,
    ),
    distiller=dict(
        teacher=dict(
            config='configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
            ckpt='data/ckpts/faster_rcnn_r50_fpn_mstrain_2x_coco_cos.pth',
        ),
        student_hooks={
            'neck': dict(type='StandardHook', path='neck'),
            ('cls', 'reg'): dict(type='MultiTensorsHook', path='bbox_head'),
        },
        student_trackings=dict(
            anchors=dict(type='StandardHook', path='bbox_head.anchors')),
        teacher_hooks=dict(
            teacher_neck=dict(type='StandardHook', path='neck')),
        adapts=dict(
            neck_adapted=dict(
                type='Conv2d',
                fields=['neck'],
                parallel=5,
                in_channels=256,
                out_channels=256,
                kernel_size=1,
            ),
            ious=dict(
                type='IoU',
                fields=['anchors', 'gt_bboxes'],
            ),
            masks=dict(
                type='FGFIMask',
                fields=['ious'],
            ),
        ),
        losses=dict(
            loss_feat=dict(
                type='FGFILoss',
                fields=['neck_adapted', 'teacher_neck', 'masks'],
                parallel=True,
                weight=3.0,
            )),
        schedulers=dict(
            warmup=dict(
                type='WarmupScheduler',
                fields=['loss_feat'],
                iter_=2000,
            ),
            early_stop=dict(
                type='EarlyStopScheduler',
                fields=['loss_feat'],
                iter_=7330 * 8,
            ),
        ),
    ))
