_base_ = [
    'retinanet_r18_fpn_mstrain_1x_coco.py',
]

model = dict(
    type='LabelEnc',
    distiller=dict(
        student_hooks=dict(
            neck=dict(
                type='StandardHook',
                path='neck',
            ),
        ),
        losses=dict(
            loss_label_enc=dict(
                type='LabelEncLoss',
                fields=['neck', 'label_neck'],
                num_channels=256,
            ),
        ),
    ),
)
