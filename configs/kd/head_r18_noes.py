_base_ = [
    'retina_fcos_faster_r18.py',
]

model = dict(
    type='HEAD',
    warmup=dict(
        warmup=dict(
            type='WarmupScheduler',
            fields=[
                'loss_cls_fcos',
                'loss_bbox_fcos',
                'loss_centerness_fcos',
                'loss_cls',
                'loss_bbox',
            ],
            iter_=2000,
            value=1.0,
        )),
    distiller=dict(
        student_hooks=dict(
            retina_cls=dict(
                type='MultiCallsHook',
                path='rpn_head.cls_convs[-1]',
            ),
            fcos_cls=dict(
                type='MultiCallsHook',
                path='rpn_head._extra_heads["fcos"].cls_convs[-1]',
            ),
            rcnn_bbox=dict(
                type='StandardHook',
                path='roi_head.bbox_head.shared_fcs[-1]',
            ),
        ),
        student_trackings=dict(
            bbox_ids=dict(type='StandardHook', path='bbox_ids')),
        adapts={
            'retina_cls_adapted':
            dict(
                type='Conv2d',
                fields=['retina_cls'],
                parallel=True,
                in_channels=256,
                out_channels=256,
                kernel_size=1,
            ),
            'fcos_cls_detached':
            dict(
                type='Detach',
                fields=['fcos_cls'],
                parallel=True,
            ),
            'retina_cls_reshaped':
            dict(
                type='Rearrange',
                fields=['retina_cls'],
                parallel=True,
                pattern='bs dim h w -> bs h w dim',
            ),
            'rcnn_bbox_detached':
            dict(
                type='Detach',
                fields=['rcnn_bbox'],
            ),
            ('rcnn_bbox_filtered', 'bbox_poses', 'anchor_ids'):
            dict(
                type='CustomAdapt',
                fields=['rcnn_bbox_detached', 'bbox_ids'],
                stride=1,
            ),
            'retina_cls_indexed':
            dict(
                type='Index',
                fields=['retina_cls_reshaped', 'bbox_poses'],
            ),
            'retina_cls_decoupled':
            dict(
                type='Decouple',
                fields=['retina_cls_indexed', 'anchor_ids'],
                num=9,
                in_features=256,
                out_features=1024,
            ),
        },
        losses=dict(
            loss_self_fcos=dict(
                type='MSELoss',
                fields=['retina_cls_adapted', 'fcos_cls_detached'],
                parallel=True,
                weight=1.0,
            ),
            loss_self_rcnn=dict(
                type='MSELoss',
                fields=['retina_cls_decoupled', 'rcnn_bbox_filtered'],
                weight=1.3,
            ),
        ),
        schedulers=dict(
            warmup=dict(
                type='WarmupScheduler',
                fields=['loss_self_fcos', 'loss_self_rcnn'],
                iter_=2000,
            ),
        ),
    ),
)
resume_from='work_dirs/head/head_r18/epoch_8.pth'
