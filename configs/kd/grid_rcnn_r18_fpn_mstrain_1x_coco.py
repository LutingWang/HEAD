_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    'coco_detection_mstrain.py',
    'schedule_1x.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='GridRCNN',
    backbone=dict(
        depth=18, init_cfg=dict(checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]),
    rpn_head=dict(
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='GridRoIHead',
        bbox_head=dict(
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            with_reg=False),
        grid_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        grid_head=dict(
            type='GridHead',
            grid_points=9,
            num_convs=8,
            in_channels=256,
            point_feat_channels=64,
            norm_cfg=dict(type='GN', num_groups=36),
            loss_grid=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=15))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(allowed_border=0),
        rpn_proposal=dict(max_per_img=2000),
        rcnn=dict(pos_radius=1, max_num_grid=192)),
    test_cfg=dict(
        rcnn=dict(score_thr=0.03, nms=dict(iou_threshold=0.3))))