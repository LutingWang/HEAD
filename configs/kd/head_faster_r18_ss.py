_base_ = [
    'retina_faster_r18_ss.py',
]

model = dict(
    type='HEAD',
    warmup=dict(
        warmup=dict(
            type='WarmupScheduler',
            tensor_names=[
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
                tensor_names=['retina_cls'],
                multilevel=True,
                pattern='bs dim h w -> bs h w dim',
            ),
            'rcnn_bbox_detached':
            dict(
                type='Detach',
                tensor_names=['rcnn_bbox'],
            ),
            ('rcnn_bbox_filtered', 'bbox_poses', 'anchor_ids'):
            dict(
                type='CustomAdapt',
                tensor_names=['rcnn_bbox_detached', 'bbox_ids'],
                stride=1,
            ),
            'retina_cls_indexed':
            dict(
                type='Index',
                tensor_names=['retina_cls_reshaped', 'bbox_poses'],
            ),
            'retina_cls_decoupled':
            dict(
                type='Decouple',
                tensor_names=['retina_cls_indexed', 'anchor_ids'],
                num=9,
                in_features=256,
                out_features=1024,
            ),
        },
        losses=dict(
            self_rcnn=dict(
                type='MSELoss',
                tensor_names=['retina_cls_decoupled', 'rcnn_bbox_filtered'],
                weight=1.3,
            )),
        schedulers=dict(
            warmup=dict(
                type='WarmupScheduler',
                tensor_names=['loss_self_rcnn'],
                iter_=2000,
            ),
            early_stop=dict(
                type='EarlyStopScheduler',
                tensor_names=['loss_self_rcnn'],
                iter_=7330 * 8,
            ),
        ),
    ),
)
