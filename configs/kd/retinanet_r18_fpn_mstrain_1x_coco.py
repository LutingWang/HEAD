_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    'coco_detection_mstrain.py',
    'schedule_1x.py',
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        depth=18, init_cfg=dict(checkpoint='pretrained/torchvision/resnet18-5c106cde.pth')),
    neck=dict(in_channels=[64, 128, 256, 512]))
