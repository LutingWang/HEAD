_base_ = [
    'retinanet_r18_fpn_mstrain_1x_coco.py',
]

model = dict(
    type='SingleTeacherSingleStageDetector',
    distiller=dict(
        teacher=dict(
            config='configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
            ckpt='data/ckpts/faster_rcnn_r50_fpn_mstrain_2x_coco_cos.pth',
        ),
        student_hooks=dict(neck=dict(type='StandardHook', path='neck')),
        teacher_hooks=dict(
            teacher_neck=dict(type='StandardHook', path='neck'),
            teacher_cls=dict(type='MultiCallsHook', path='rpn_head.rpn_cls'),
        ),
        adapts=dict(
            neck_adapted=dict(
                type='Conv2d',
                fields=['neck'],
                parallel=True,
                in_channels=256,
                out_channels=256,
                kernel_size=1,
            ),
            mask=dict(
                type='FRSMask',
                fields=['teacher_cls'],
                parallel=True,
                with_logits=True,
            ),
        ),
        losses=dict(
            loss_feat=dict(
                type='MSE2DLoss',
                fields=['neck_adapted', 'teacher_neck', 'mask'],
                parallel=True,
                weight=1.0,
            )),
    ))
