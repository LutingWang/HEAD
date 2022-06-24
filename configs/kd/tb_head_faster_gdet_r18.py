_base_ = [
    'retina_faster_r18.py',
]

model = dict(
    type='TB_HEAD',
    warmup=dict(
        warmup=dict(
            type='WarmupScheduler',
            fields=[
                'loss_cls',
                'loss_bbox',
            ],
            iter_=2000,
            value=1.0,
        )),
    distiller=dict(
        teacher=dict(
            config='configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
            ckpt='data/ckpts/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth',
        ),
        weight_transfer={
            'student.roi_head': 'teacher.roi_head',
        },
        student_hooks=dict(
            neck=dict(type='StandardHook', path='neck'),
            cls=dict(type='MultiCallsHook', path='rpn_head.cls_convs[-1]'),
            roi_feats=dict(
                type='StandardHook',
                path='roi_head.bbox_roi_extractor',
            ),
            rcnn_bbox_aux=dict(
                type='StandardHook',
                path='roi_head.bbox_head.shared_fcs[0]',
            ),
            rcnn_bbox=dict(
                type='StandardHook',
                path='roi_head.bbox_head.shared_fcs[-1]',
            ),
        ),
        teacher_hooks=dict(
            teacher_roi_feats=dict(
                type='StandardHook',
                path='roi_head.bbox_roi_extractor',
            ),
            teacher_rcnn_bbox_aux=dict(
                type='StandardHook',
                path='roi_head.bbox_head.shared_fcs[0]',
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
                fields=['cls'],
                parallel=True,
                pattern='bs dim h w -> bs h w dim',
            ),
            ('teacher_rcnn_bbox_filtered', 'bbox_poses', 'anchor_ids'):
            dict(
                type='CustomAdapt',
                fields=['teacher_rcnn_bbox', 'bbox_ids'],
                stride=1,
            ),
            'cls_indexed':
            dict(
                type='Index',
                fields=['cls_reshaped', 'bbox_poses'],
            ),
            'cls_decoupled':
            dict(
                type='Decouple',
                fields=['cls_indexed', 'anchor_ids'],
                num=9,
                in_features=256,
                out_features=1024,
                bias=False,
            ),
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
            'rcnn_bbox_aux_adapted':
            dict(
                type='Linear',
                fields=['rcnn_bbox_aux'],
                in_features=1024,
                out_features=1024,
            ),
            'rcnn_bbox_adapted':
            dict(
                type='Linear',
                fields=['rcnn_bbox'],
                in_features=1024,
                out_features=1024,
            ),
        },
        losses=dict(
            loss_ckd=dict(
                type='CKDLoss',
                fields=[
                    'cls_decoupled', 'teacher_rcnn_bbox_filtered',
                ],
                weight=0.5,
            ),
            loss_sgfi=dict(
                type='SGFILoss',
                fields=['roi_feats_adapted', 'teacher_roi_feats'],
                weight=1.0,
            ),
            loss_mimic_rcnn_aux=dict(
                type='MSELoss',
                fields=['rcnn_bbox_aux_adapted', 'teacher_rcnn_bbox_aux'],
                weight=2.0,
            ),
            loss_mimic_rcnn=dict(
                type='MSELoss',
                fields=['rcnn_bbox_adapted', 'teacher_rcnn_bbox'],
                weight=2.0,
            ),
        ),
        schedulers=dict(
            warmup=dict(
                type='WarmupScheduler',
                fields=[
                    'loss_mimic_rcnn', 'loss_mimic_rcnn_aux',
                ],
                iter_=2000,
            ),
        ),
    ),
)
