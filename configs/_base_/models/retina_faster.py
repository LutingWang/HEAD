_base_ = [
    'faster_rcnn.py',
]

model = dict(
    type='CrossStageDetector',
    neck=dict(
        start_level=1,
        add_extra_convs='on_input',
    ),
    rpn_head=dict(
        _delete_=True,
        type='RetinaRPNHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),
    roi_head=dict(
        type='StandardRoIHeadWithBBoxIDs',
        bbox_roi_extractor=dict(featmap_strides=[8, 16, 32, 64]),
    ),
    train_cfg=dict(
        rpn=dict(
            _delete_=True,
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            _delete_=True,
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
        ),
    ),
)
