_base_ = [
    '../retinanet/retinanet_r50_fpn_1x_coco.py',
]

model = dict(
    type='SingleTeacherSingleStageDetector',
    bbox_head=dict(
        type='RetinaMultiHead',
        cache_anchors=True,
    ),
    distiller=dict(
        teacher=dict(
            config=('configs/retinanet/retinanet_x101_64x4d_fpn_'
                    'mstrain_640-800_3x_coco.py'),
            ckpt=('data/ckpts/retinanet_x101_64x4d_fpn_mstrain_'
                  '3x_coco_20210719_051838-022c2187.pth'),
        ),
        student_hooks=dict(neck=dict(type='StandardHook', path='neck')),
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
                weight=1.0,
            )),
    ))
