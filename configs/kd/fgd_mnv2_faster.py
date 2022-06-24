_base_ = [
    'faster_rcnn_mnv2_fpn_mstrain_1x_coco.py',
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
                strides=[4, 8, 16, 32, 64],
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
                weight=1e-4,
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
                weight=1.0,
            ),
        ),
        schedulers=dict(
            warmup=dict(
                type='WarmupScheduler',
                fields=[
                    'loss_feat', 'loss_attn_spatial', 'loss_attn_channel',
                    'loss_global_'
                ],
                iter_=2000,
            )),
    ))
