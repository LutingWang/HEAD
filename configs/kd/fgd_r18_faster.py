_base_ = [
    'faster_rcnn_r18_fpn_mstrain_1x_coco.py',
]

model = dict(
    type='SingleTeacherTwoStageDetector',
    distiller=dict(
        teacher=dict(
            # config='configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
            # ckpt=('data/ckpts/faster_rcnn_r50_fpn_mstrain_3x_coco_'
            #      '20210524_110822-e10bd31c.pth'),
            # config='configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py',
            # ckpt=('data/ckpts/reppoints_moment_r50_fpn_gn-neck+head_2x_'
            #       'coco_20200329-91babaa2.pth'),
            config='configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py',
            ckpt='data/ckpts/cascade_rcnn_r50_fpn_mstrain_3x_coco.pth',
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
                strides=[4, 8, 16, 32, 64],
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
                weight=1e-4,
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
                weight=1.0,
            ),
        ),
        schedulers=dict(
            warmup=dict(
                type='WarmupScheduler',
                tensor_names=[
                    'loss_feat', 'loss_attn_spatial', 'loss_attn_channel',
                    'loss_global_'
                ],
                iter_=2000,
            )),
    ))
