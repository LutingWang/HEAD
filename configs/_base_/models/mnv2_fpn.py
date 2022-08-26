model = dict(
    backbone=dict(
        type='MobileNetV2',
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=(
                'pretrained/mmdetection/'
                'mobilenet_v2_batch256_imagenet-ff34753d.pth'
            ),
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 1280],
        out_channels=256,
        num_outs=5,
    ),
)
