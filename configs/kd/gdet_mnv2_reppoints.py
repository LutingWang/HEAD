_base_ = [
    'fcos_mnv2_fpn_mstrain_1x_coco.py',
]

model = dict(
    type='GDetSingleStage',
    bbox_head=dict(type='FCOSRPNHead'),
    distiller=dict(
        teacher=dict(
            config='configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py',
            ckpt=('data/ckpts/reppoints_moment_r50_fpn_gn-neck+head_2x_'
                  'coco_20200329-91babaa2.pth'),
        ),
        student_hooks=dict(
            neck=dict(type='StandardHook', path='neck'),
            teacher_roi_feats=dict(
                type='StandardHook',
                path='roi_head.bbox_roi_extractor',
            ),
        ),
        adapts={
            'roi_feats':
            dict(
                type='RoIAlign',
                tensor_names=['neck', 'bboxes'],
                strides=[8, 16, 32, 64, 128],
            ),
            'roi_feats_adapted':
            dict(
                type='Conv2d',
                tensor_names=['roi_feats'],
                multilevel=True,
                in_channels=256,
                out_channels=256,
                kernel_size=1,
            ),
        },
        losses=dict(
            sgfi=dict(
                type='SGFILoss',
                tensor_names=['roi_feats_adapted', 'teacher_roi_feats'],
                weight=1.0,
            ),
        ),
    ),
)
