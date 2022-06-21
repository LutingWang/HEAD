_base_ = [
    'retinanet_r18_fpn_mstrain_1x_coco.py',
]

model = dict(
    type='MultiHeadSingleStageDetector',
    bbox_head=dict(
        type='RetinaMultiHead',
        extra_heads=dict(
            fcos=dict(
                type='FCOSHead',
                # norm_on_bbox=True,
                # centerness_on_reg=True,
                # dcn_on_last_conv=False,
                # center_sampling=True,
                # conv_bias=True,
                num_classes=80,
                in_channels=256,
                stacked_convs=4,
                feat_channels=256,
                strides=[8, 16, 32, 64, 128],
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
                loss_centerness=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0,
                ),
                train_cfg=dict(
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
                test_cfg=dict(
                    nms_pre=1000,
                    min_bbox_size=0,
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100,
                ),
            )),
    ),
)
