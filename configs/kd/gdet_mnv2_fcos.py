_base_ = [
    'fcos_mnv2_fpn_mstrain_1x_coco.py',
]

model = dict(
    type='GDet',
    bbox_head=dict(type='FCOSRPNHead'),
    distiller=dict(
        teacher=dict(
            config='configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
            config_options={
                'model.train_cfg.rcnn.sampler.add_gt_as_proposals': False
            },
            ckpt=('data/ckpts/faster_rcnn_r50_fpn_mstrain_3x_coco_'
                 '20210524_110822-e10bd31c.pth'),
        ),
        student_hooks=dict(
            neck=dict(type='StandardHook', path='neck'),
            # cls=dict(type='MultiCallsHook', path='bbox_head.cls_convs[-1]'),
        ),
        teacher_hooks=dict(
            teacher_roi_feats=dict(
                type='StandardHook',
                path='roi_head.bbox_roi_extractor',
            ),
            # teacher_rcnn_bbox=dict(
            #     type='StandardHook',
            #     path='roi_head.bbox_head.shared_fcs[-1]',
            # ),
        ),
        adapts={
            # 'cls_reshaped':
            # dict(
            #     type='Rearrange',
            #     fields=['cls'],
            #     parallel=True,
            #     pattern='bs dim h w -> bs h w dim',
            # ),
            # ('teacher_rcnn_bbox_filtered', 'bbox_poses', 'anchor_ids'):
            # dict(
            #     type='CustomAdapt',
            #     fields=['teacher_rcnn_bbox', 'bbox_ids'],
            #     stride=1,
            # ),
            # 'cls_indexed':
            # dict(
            #     type='Index',
            #     fields=['cls_reshaped', 'bbox_poses'],
            # ),
            # 'cls_decoupled':
            # dict(
            #     type='Linear',
            #     fields=['cls_indexed'],
            #     in_features=256,
            #     out_features=1024,
            #     bias=False,
            # ),
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
            # loss_ckd=dict(
            #     type='CKDLoss',
            #     fields=[
            #         'cls_decoupled', 'teacher_rcnn_bbox_filtered', 'bboxes',
            #     ],
            #     weight=0.5,
            # ),
            loss_sgfi=dict(
                type='SGFILoss',
                fields=['roi_feats_adapted', 'teacher_roi_feats'],
                weight=1.0,
            ),
        ),
    ),
)
