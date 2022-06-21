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
        type='DynamicRoIHead',
        bbox_head=dict(
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn_proposal=dict(nms=dict(iou_threshold=0.85)),
        rcnn=dict(
            dynamic_rcnn=dict(
                iou_topk=75,
                beta_topk=10,
                update_iter_interval=100,
                initial_iou=0.4,
                initial_beta=1.0))),
    test_cfg=dict(rpn=dict(nms=dict(iou_threshold=0.85))))
