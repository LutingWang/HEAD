_base_ = [
    'fcos_retina_faster_mnv2.py',
]

model = dict(
    type='HEAD',
    warmup=dict(
        warmup=dict(
            type='WarmupScheduler',
            tensor_names=[
                'loss_cls_retina',
                'loss_bbox_retina',
                'loss_centerness_retina',
                'loss_cls',
                'loss_bbox',
            ],
            iter_=2000,
            value=1.0,
        )),
    distiller=dict(
        student_hooks=dict(
            fcos_cls=dict(
                type='MultiCallsHook',
                path='rpn_head.cls_convs[-1]',
            ),
            retina_cls=dict(
                type='MultiCallsHook',
                path='rpn_head._extra_heads["retina"].cls_convs[-1]',
            ),
            rcnn_bbox=dict(
                type='StandardHook',
                path='roi_head.bbox_head.shared_fcs[-1]',
            ),
        ),
        student_trackings=dict(
            bbox_ids=dict(type='StandardHook', path='bbox_ids')),
        adapts={
            'fcos_cls_adapted':
            dict(
                type='Conv2d',
                tensor_names=['fcos_cls'],
                multilevel=True,
                in_channels=256,
                out_channels=256,
                kernel_size=1,
            ),
            'retina_cls_detached':
            dict(
                type='Detach',
                tensor_names=['retina_cls'],
                multilevel=True,
            ),
            'fcos_cls_reshaped':
            dict(
                type='Rearrange',
                tensor_names=['fcos_cls'],
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
            'fcos_cls_indexed':
            dict(
                type='Index',
                tensor_names=['fcos_cls_reshaped', 'bbox_poses'],
            ),
            'fcos_cls_decoupled':
            dict(
                type='Linear',
                tensor_names=['fcos_cls_indexed'],
                in_features=256,
                out_features=1024,
            ),
        },
        losses=dict(
            self_retina=dict(
                type='MSELoss',
                tensor_names=['fcos_cls_adapted', 'retina_cls_detached'],
                multilevel=True,
                weight=1.0,
            ),
            self_rcnn=dict(
                type='MSELoss',
                tensor_names=['fcos_cls_decoupled', 'rcnn_bbox_filtered'],
                weight=1.3,
            ),
        ),
        schedulers=dict(
            warmup=dict(
                type='WarmupScheduler',
                tensor_names=['loss_self_retina', 'loss_self_rcnn'],
                iter_=2000,
            ),
            early_stop=dict(
                type='EarlyStopScheduler',
                tensor_names=['loss_self_retina', 'loss_self_rcnn'],
                iter_=7330 * 8,
            ),
        ),
    ),
)
