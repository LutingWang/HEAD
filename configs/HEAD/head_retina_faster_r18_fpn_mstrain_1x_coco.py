_base_ = [
    '../HEAD_dag/head_dag_retina_faster_r18_fpn_mstrain_1x_coco.py',
]

model = dict(
    distiller=dict(
        student_hooks=dict(
            retina_cls=dict(
                type='MultiCallsHook',
                path='.rpn_head.cls_convs[-1]',
            ),
        ),
        adapts={
            'retina_cls_reshaped': dict(
                type='Rearrange',
                fields=['retina_cls'],
                parallel=True,
                pattern='bs dim h w -> bs h w dim',
            ),
            ('teacher_rcnn_bbox_filtered', 'bbox_poses', 'anchor_ids'): dict(
                type='CustomAdapt',
                fields=['teacher_rcnn_bbox', 'bbox_ids'],
                stride=1,
            ),
            'retina_cls_indexed': dict(
                type='Index',
                fields=['retina_cls_reshaped', 'bbox_poses'],
            ),
            'retina_cls_decoupled': dict(
                type='Linear',
                fields=['retina_cls_indexed'],
                in_features=256,
                out_features=1024,
            ),
        },
        losses=dict(
            loss_cross=dict(
                type='MSELoss',
                fields=['retina_cls_decoupled', 'teacher_rcnn_bbox_filtered'],
                weight=dict(
                    type='WarmupScheduler',
                    value=2.0,
                    iter_=2000,
                ),
            ),
        ),
    ),
)
