_base_ = [
    'retinanet_r18_fpn_mstrain_1x_coco.py',
]

model = dict(
    type='MultiHeadSingleStageDetector',
    bbox_head=dict(
        type='RetinaMultiHead',
        extra_heads=dict(
            tood=dict(
                type='TOODHead',
                num_classes=80,
                in_channels=256,
                stacked_convs=6,
                feat_channels=256,
                anchor_type='anchor_free',
                anchor_generator=dict(
                    type='AnchorGenerator',
                    ratios=[1.0],
                    octave_base_scale=8,
                    scales_per_octave=1,
                    strides=[8, 16, 32, 64, 128]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[.0, .0, .0, .0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                initial_loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    activated=True,  # use probability instead of logit as input
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_cls=dict(
                    type='QualityFocalLoss',
                    use_sigmoid=True,
                    activated=True,  # use probability instead of logit as input
                    beta=2.0,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                train_cfg=dict(
                    initial_epoch=4,
                    initial_assigner=dict(type='ATSSAssigner', topk=9),
                    assigner=dict(type='TaskAlignedAssigner', topk=13),
                    alpha=1,
                    beta=6,
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False),
                test_cfg=dict(
                    nms_pre=1000,
                    min_bbox_size=0,
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.6),
                    max_per_img=100),
            )),
    ),
)
