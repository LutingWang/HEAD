_base_ = [
    'retina_faster_ss.py',
]

model = dict(
    type='HEAD',
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
        student_hooks=dict(
            retina_cls=dict(
                type='MultiCallsHook',
                path='rpn_head.cls_convs[-1]',
            ),
            rcnn_bbox=dict(
                type='StandardHook',
                path='roi_head.bbox_head.shared_fcs[-1]',
            ),
        ),
        student_trackings=dict(
            bbox_ids=dict(type='StandardHook', path='bbox_ids')),
        adapts={
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
            loss_self_rcnn=dict(
                type='MSELoss',
                fields=['retina_cls_decoupled', 'rcnn_bbox_filtered'],
                weight=1.3,
            )),
        schedulers=dict(
            warmup=dict(
                type='WarmupScheduler',
                fields=['loss_self_rcnn'],
                iter_=2000,
            ),
            early_stop=dict(
                type='EarlyStopScheduler',
                fields=['loss_self_rcnn'],
                iter_=7330 * 8,
            ),
        ),
    ),
)
