_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    'coco_detection_mstrain.py',
    'schedule_1x.py',
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        depth=18, init_cfg=dict(checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]),
    roi_head=dict(
        type='DoubleHeadRoIHead',
        reg_roi_scale_factor=1.3,
        bbox_head=dict(
            _delete_=True,
            type='DoubleConvFCBBoxHead',
            num_convs=4,
            num_fcs=2,
            in_channels=256,
            conv_out_channels=1024,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0))))
