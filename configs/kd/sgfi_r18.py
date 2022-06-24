_base_ = [
    'fcos_r18_fpn_mstrain_1x_coco.py',
]

model = dict(
    type='GDet',
    with_ckd=False,
    bbox_head=dict(type='FCOSRPNHead'),
    distiller=dict(
        teacher=dict(
            config='configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
            config_options={
                'model.train_cfg.rcnn.sampler.add_gt_as_proposals': False
            },
            ckpt='data/ckpts/faster_rcnn_r50_fpn_mstrain_2x_coco_cos.pth',
        ),
        student_hooks=dict(
            neck=dict(type='StandardHook', path='neck')),
        teacher_hooks=dict(
            teacher_roi_feats=dict(
                type='StandardHook',
                path='roi_head.bbox_roi_extractor',
            ),
        ),
        adapts={
            'roi_feats':
            dict(
                type='RoIAlign',
                fields=['neck', 'bboxes'],
                strides=[8, 16, 32, 64, 128],
            ),
            'roi_feats_adapted':
            dict(
                type='Conv2d',
                fields=['roi_feats'],
                parallel=True,
                in_channels=256,
                out_channels=256,
                kernel_size=1,
            ),
        },
        losses=dict(
            loss_sgfi=dict(
                type='SGFILoss',
                fields=['roi_feats_adapted', 'teacher_roi_feats'],
                weight=1.0,
            ),
        ),
    ),
)
