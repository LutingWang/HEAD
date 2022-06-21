_base_ = [
    'fcos_mnv2_fpn_mstrain_1x_coco.py',
]

model = dict(
    type='SingleTeacherSingleStageDetector',
    distiller=dict(
        teacher=dict(
            # config='configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
            # ckpt=('data/ckpts/faster_rcnn_r50_fpn_mstrain_3x_coco_'
            #      '20210524_110822-e10bd31c.pth'),
            config='configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py',
            ckpt=('data/ckpts/reppoints_moment_r50_fpn_gn-neck+head_2x_'
                  'coco_20200329-91babaa2.pth'),
        ),
        student_hooks=dict(neck=dict(type='StandardHook', path='neck')),
        teacher_hooks=dict(
            teacher_neck=dict(type='StandardHook', path='neck')),
        adapts=dict(
            neck_adapted=dict(
                type='Conv2d',
                tensor_names=['neck'],
                multilevel=5,
                in_channels=256,
                out_channels=256,
                kernel_size=1,
            )),
        losses=dict(
            feat=dict(
                type='MSELoss',
                tensor_names=['neck_adapted', 'teacher_neck'],
                multilevel=True,
                weight=1.0,
            )),
    ))
