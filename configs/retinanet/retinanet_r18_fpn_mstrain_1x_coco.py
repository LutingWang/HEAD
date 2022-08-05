_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection_mstrain.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
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
optimizer = dict(lr=0.01)
