_base_ = [
    'retinanet.py',
]

model = dict(
    type='SingleTeacherSingleStageDistiller',
    distiller=dict(
        teacher=dict(
            config=('configs/retinanet/retinanet_r50_fpn_1x_coco.py'),
            ckpt=(
                'pretrained/mmdetection/retinanet_r50_fpn_mstrain_3x_coco_'
                '20210718_220633-88476508.pth'
            ),
        ),
        weight_transfer={
            '.student.bbox_head': '.teacher.bbox_head',
        },
        student_hooks=dict(neck=dict(type='StandardHook', path='neck')),
        teacher_hooks=dict(
            teacher_neck=dict(type='StandardHook', path='neck'),
        ),
        adapts=dict(
            neck_adapted=dict(
                type='Conv2d',
                fields=['neck'],
                parallel=5,
                in_channels=256,
                out_channels=256,
                kernel_size=1,
            ),
        ),
        losses=dict(
            loss_feat=dict(
                type='MSELoss',
                fields=['neck_adapted', 'teacher_neck'],
                parallel=True,
                weight=1.0,
            )
        ),
    )
)
