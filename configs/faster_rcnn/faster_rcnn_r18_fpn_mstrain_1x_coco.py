_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection_mstrain.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        depth=18, init_cfg=dict(checkpoint='pretrained/torchvision/resnet18-5c106cde.pth')),
    neck=dict(in_channels=[64, 128, 256, 512]))
