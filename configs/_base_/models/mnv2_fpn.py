model = dict(
    backbone=dict(
        _delete_=True,
        type='MobileNetV2',
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://mmdet/mobilenet_v2',
        ),
    ),
    neck=dict(in_channels=[24, 32, 96, 1280]),
)
