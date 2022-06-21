_base_ = [
    'retinanet_r18_fpn_mstrain_1x_coco.py',
]

model = dict(
    type='MultiHeadSingleStageDetector',
    bbox_head=dict(
        type='RetinaMultiHead',
        extra_heads=dict(
            paa=dict(
                type='PAAHead',
                reg_decoded_bbox=True,
                score_voting=True,
                topk=9,
                num_classes=80,
                in_channels=256,
                stacked_convs=4,
                feat_channels=256,
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
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=1.3),
                loss_centerness=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5),
                train_cfg=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.1,
                        neg_iou_thr=0.1,
                        min_pos_iou=0,
                        ignore_iof_thr=-1),
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
