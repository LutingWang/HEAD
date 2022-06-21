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
                tensor_names=['neck'],
                multilevel=True,
                temperature=0.5,
            ),
            teacher_attn_spatial=dict(
                type='AbsMeanSpatialAttention',
                tensor_names=['teacher_neck'],
                multilevel=True,
                temperature=0.5,
            ),
            attn_channel=dict(
                type='AbsMeanChannelAttention',
                tensor_names=['neck'],
                multilevel=True,
                temperature=0.5,
            ),
            teacher_attn_channel=dict(
                type='AbsMeanChannelAttention',
                tensor_names=['teacher_neck'],
                multilevel=True,
                temperature=0.5,
            ),
            masks=dict(
                type='FGDMask',
                tensor_names=['batch_input_shape', 'gt_bboxes'],
                neg_gain=0.5,
                strides=[8, 16, 32, 64, 128],
                ceil_mode=True,
            ),
            global_=dict(
                type='ContextBlock',
                tensor_names=['neck'],
                multilevel=5,
                in_channels=256,
                ratio=0.5,
            ),
            teacher_global=dict(
                type='ContextBlock',
                tensor_names=['teacher_neck'],
                multilevel=5,
                in_channels=256,
                ratio=0.5,
            ),
        ),
        losses=dict(
            feat=dict(
                type='FGDLoss',
                tensor_names=[
                    'neck', 'teacher_neck', 'teacher_attn_spatial',
                    'teacher_attn_channel', 'masks'
                ],
                multilevel=True,
                weight=5e-4,
                reduction='sum',
            ),
            attn_spatial=dict(
                type='L1Loss',
                tensor_names=['attn_spatial', 'teacher_attn_spatial'],
                multilevel=True,
                weight=2.5e-4,
                reduction='sum',
            ),
            attn_channel=dict(
                type='L1Loss',
                tensor_names=['attn_channel', 'teacher_attn_channel'],
                multilevel=True,
                weight=2.5e-4,
                reduction='sum',
            ),
            global_=dict(
                type='MSELoss',
                tensor_names=['global_', 'teacher_global'],
                multilevel=True,
                weight=2.5e-6,
                reduction='sum',
            ),
        ),
    ))
