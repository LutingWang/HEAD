_base_ = [
    '../retinanet/retinanet_r50_fpn_2x_coco.py',
]

model = dict(
    type='SingleTeacherSingleStageDetector',
    distiller=dict(
        teacher=dict(
            config=('configs/retinanet/'
                    'retinanet_x101_64x4d_fpn_mstrain_640-800_3x_coco.py'),
            ckpt=('data/ckpts/retinanet_x101_64x4d_fpn_'
                  'mstrain_3x_coco_20210719_051838-022c2187.pth'),
        ),
        student_hooks=dict(neck=dict(type='StandardHook', path='neck')),
        teacher_hooks=dict(
            teacher_neck=dict(type='StandardHook', path='neck')),
        adapts=dict(
            attn_spatial=dict(
                type='AbsMeanSpatialAttention',
                fields=['neck'],
                parallel=True,
                temperature=0.5,
            ),
            teacher_attn_spatial=dict(
                type='AbsMeanSpatialAttention',
                fields=['teacher_neck'],
                parallel=True,
                temperature=0.5,
            ),
            attn_channel=dict(
                type='AbsMeanChannelAttention',
                fields=['neck'],
                parallel=True,
                temperature=0.5,
            ),
            teacher_attn_channel=dict(
                type='AbsMeanChannelAttention',
                fields=['teacher_neck'],
                parallel=True,
                temperature=0.5,
            ),
            masks=dict(
                type='FGDMask',
                fields=['batch_input_shape', 'gt_bboxes'],
                neg_gain=0.5,
                strides=[8, 16, 32, 64, 128],
                ceil_mode=True,
            ),
            global_=dict(
                type='ContextBlock',
                fields=['neck'],
                parallel=5,
                in_channels=256,
                ratio=0.5,
            ),
            teacher_global=dict(
                type='ContextBlock',
                fields=['teacher_neck'],
                parallel=5,
                in_channels=256,
                ratio=0.5,
            ),
        ),
        losses=dict(
            loss_feat=dict(
                type='FGDLoss',
                fields=[
                    'neck', 'teacher_neck', 'teacher_attn_spatial',
                    'teacher_attn_channel', 'masks'
                ],
                parallel=True,
                weight=5e-4,
                reduction='sum',
            ),
            loss_attn_spatial=dict(
                type='L1Loss',
                fields=['attn_spatial', 'teacher_attn_spatial'],
                parallel=True,
                weight=2.5e-4,
                reduction='sum',
            ),
            loss_attn_channel=dict(
                type='L1Loss',
                fields=['attn_channel', 'teacher_attn_channel'],
                parallel=True,
                weight=2.5e-4,
                reduction='sum',
            ),
            loss_global_=dict(
                type='MSELoss',
                fields=['global_', 'teacher_global'],
                parallel=True,
                weight=2.5e-6,
                reduction='sum',
            ),
        ),
    ))
