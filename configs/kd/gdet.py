_base_ = [
    'retinanet_r50_fpn_mstrain_1x_coco.py',
]

model = dict(
    type='GDet',
    bbox_head=dict(type='RetinaRPNHead'),
    distiller=dict(
        teacher=dict(
            config=('configs/faster_rcnn/'
                    'faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py'),
            config_options={
                'model.train_cfg.rcnn.sampler.add_gt_as_proposals': False
            },
            ckpt=('data/ckpts/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco_'
                  '20210524_124528-26c63de6.pth'),
        ),
        student_hooks=dict(
            neck=dict(type='StandardHook', path='neck'),
            cls=dict(type='MultiCallsHook', path='bbox_head.cls_convs[-1]')),
        teacher_hooks=dict(
            teacher_roi_feats=dict(
                type='StandardHook',
                path='roi_head.bbox_roi_extractor',
            ),
            teacher_rcnn_bbox=dict(
                type='StandardHook',
                path='roi_head.bbox_head.shared_fcs[-1]',
            ),
        ),
        adapts={
            'cls_reshaped':
            dict(
                type='Rearrange',
                tensor_names=['cls'],
                multilevel=True,
                pattern='bs dim h w -> bs h w dim',
            ),
            ('teacher_rcnn_bbox_filtered', 'bbox_poses', 'anchor_ids'):
            dict(
                type='CustomAdapt',
                tensor_names=['teacher_rcnn_bbox', 'bbox_ids'],
                stride=1,
            ),
            'cls_indexed':
            dict(
                type='Index',
                tensor_names=['cls_reshaped', 'bbox_poses'],
            ),
            'cls_decoupled':
            dict(
                type='Decouple',
                tensor_names=['cls_indexed', 'anchor_ids'],
                num=9,
                in_features=256,
                out_features=1024,
                bias=False,
            ),
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
            ckd=dict(
                type='CKDLoss',
                tensor_names=[
                    'cls_decoupled', 'teacher_rcnn_bbox_filtered', 'bboxes',
                ],
                weight=0.5,
            ),
            sgfi=dict(
                type='SGFILoss',
                tensor_names=['roi_feats_adapted', 'teacher_roi_feats'],
                weight=1.0,
            ),
        ),
    ),
)
