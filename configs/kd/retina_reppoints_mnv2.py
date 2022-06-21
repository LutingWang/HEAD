_base_ = [
    'retinanet_mnv2_fpn_mstrain_1x_coco.py',
]

model = dict(
    type='MultiHeadSingleStageDetector',
    bbox_head=dict(
        type='RetinaMultiHead',
        extra_heads=dict(
            reppoints=dict(
                type='RepPointsHead',
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                num_classes=80,
                in_channels=256,
                feat_channels=256,
                point_feat_channels=256,
                stacked_convs=3,
                num_points=9,
                gradient_mul=0.1,
                point_strides=[8, 16, 32, 64, 128],
                point_base_scale=4,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
                loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
                transform_method='moment',
                train_cfg=dict(
                    init=dict(
                        assigner=dict(type='PointAssigner', scale=4, pos_num=1),
                        allowed_border=-1,
                        pos_weight=-1,
                        debug=False),
                    refine=dict(
                        assigner=dict(
                            type='MaxIoUAssigner',
                            pos_iou_thr=0.5,
                            neg_iou_thr=0.4,
                            min_pos_iou=0,
                            ignore_iof_thr=-1),
                        allowed_border=-1,
                        pos_weight=-1,
                        debug=False)),
                test_cfg=dict(
                    nms_pre=1000,
                    min_bbox_size=0,
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100),
            )),
    ),
)
