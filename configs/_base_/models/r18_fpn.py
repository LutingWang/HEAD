_base_ = [
    'r50_fpn.py',
]

model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(
            checkpoint='pretrained/torchvision/resnet18-f37072fd.pth',
        ),
    ),
    neck=dict(in_channels=[64, 128, 256, 512]),
)
