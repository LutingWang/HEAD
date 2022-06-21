_base_ = [
    '../retinanet/retinanet_r50_fpn_1x_coco.py',
]

model = dict(
    type='SingleTeacherSingleStageDetector',
    distiller=dict(
        teacher=dict(
            config=('configs/retinanet/retinanet_x101_64x4d_fpn_'
                    'mstrain_640-800_3x_coco.py'),
            ckpt=('data/ckpts/retinanet_x101_64x4d_fpn_mstrain_'
                  '3x_coco_20210719_051838-022c2187.pth'),
        ),
        student_hooks=dict(neck=dict(type='StandardHook', path='neck')),
        teacher_hooks=dict(
            teacher_neck=dict(type='StandardHook', path='neck')),
        adapts=dict(
            neck=dict(
                type='Conv2d',
                tensor_names=['neck'],
                multilevel=5,
                in_channels=256,
                out_channels=256,
                kernel_size=1,
            ),
            masks=dict(
                type='DeFeatMask',
                tensor_names=['batch_input_shape', 'gt_bboxes'],
                neg_gain=4,
                strides=[8, 16, 32, 64, 128],
                ceil_mode=True,
            ),
        ),
        losses=dict(
            neck=dict(
                type='MSELoss',
                tensor_names=['neck', 'teacher_neck', 'masks'],
                multilevel=True,
                weight=1 / 256,
                reduction='sum',
            )),
    ))
