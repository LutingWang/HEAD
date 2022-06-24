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
            neck=dict(
                type='Conv2d',
                fields=['neck'],
                parallel=5,
                in_channels=256,
                out_channels=256,
                kernel_size=1,
            ),
            masks=dict(
                type='DeFeatMask',
                fields=['batch_input_shape', 'gt_bboxes'],
                neg_gain=4,
                strides=[4, 8, 16, 32, 64],
                ceil_mode=True,
            ),
        ),
        losses=dict(
            loss_neck=dict(
                type='MSELoss',
                fields=['neck', 'teacher_neck', 'masks'],
                parallel=True,
                weight=1 / 1024,
                reduction='sum',
            )),
        schedulers=dict(
            warmup=dict(
                type='WarmupScheduler', fields=['loss_neck'],
                iter_=2000))))
