_base_ = [
    'faster_rcnn_mnv2_fpn_mstrain_1x_coco.py',
]

model = dict(
    type='TB_HEADTwoStage',
    distiller=dict(
        teacher=dict(
            config='configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py',
            ckpt='data/ckpts/cascade_rcnn_r50_fpn_mstrain_3x_coco.pth',
        ),
        weight_transfer={
            'student.cascade_roi_head': 'teacher.roi_head',
        },
        student_hooks=dict(
            rcnn_bbox=dict(
                type='StandardHook',
                path='roi_head.bbox_head.shared_fcs[-1]',
            ),
            rcnn_bbox0=dict(
                type='StandardHook',
                path='cascade_roi_head.bbox_head[0].shared_fcs[-1]',
            ),
            rcnn_bbox_aux0=dict(
                type='StandardHook',
                path='cascade_roi_head.bbox_head[0].shared_fcs[0]',
            ),
            rcnn_bbox1=dict(
                type='StandardHook',
                path='cascade_roi_head.bbox_head[1].shared_fcs[-1]',
            ),
            rcnn_bbox_aux1=dict(
                type='StandardHook',
                path='cascade_roi_head.bbox_head[1].shared_fcs[0]',
            ),
            rcnn_bbox2=dict(
                type='StandardHook',
                path='cascade_roi_head.bbox_head[2].shared_fcs[-1]',
            ),
            rcnn_bbox_aux2=dict(
                type='StandardHook',
                path='cascade_roi_head.bbox_head[2].shared_fcs[0]',
            ),
        ),
        teacher_hooks=dict(
            teacher_rcnn_bbox0=dict(
                type='StandardHook',
                path='roi_head.bbox_head[0].shared_fcs[-1]',
            ),
            teacher_rcnn_bbox_aux0=dict(
                type='StandardHook',
                path='roi_head.bbox_head[0].shared_fcs[0]',
            ),
            teacher_rcnn_bbox1=dict(
                type='StandardHook',
                path='roi_head.bbox_head[1].shared_fcs[-1]',
            ),
            teacher_rcnn_bbox_aux1=dict(
                type='StandardHook',
                path='roi_head.bbox_head[1].shared_fcs[0]',
            ),
            teacher_rcnn_bbox2=dict(
                type='StandardHook',
                path='roi_head.bbox_head[2].shared_fcs[-1]',
            ),
            teacher_rcnn_bbox_aux2=dict(
                type='StandardHook',
                path='roi_head.bbox_head[2].shared_fcs[0]',
            ),
        ),
        adapts={
            'rcnn_bbox_adapted':
            dict(
                type='Linear',
                tensor_names=['rcnn_bbox'],
                in_features=1024,
                out_features=1024,
                bias=False,
            ),
            'rcnn_bbox_aux0_adapted':
            dict(
                type='Linear',
                tensor_names=['rcnn_bbox_aux0'],
                in_features=1024,
                out_features=1024,
            ),
            'rcnn_bbox0_adapted':
            dict(
                type='Linear',
                tensor_names=['rcnn_bbox0'],
                in_features=1024,
                out_features=1024,
            ),
            'rcnn_bbox_aux1_adapted':
            dict(
                type='Linear',
                tensor_names=['rcnn_bbox_aux1'],
                in_features=1024,
                out_features=1024,
            ),
            'rcnn_bbox1_adapted':
            dict(
                type='Linear',
                tensor_names=['rcnn_bbox1'],
                in_features=1024,
                out_features=1024,
            ),
            'rcnn_bbox_aux2_adapted':
            dict(
                type='Linear',
                tensor_names=['rcnn_bbox_aux2'],
                in_features=1024,
                out_features=1024,
            ),
            'rcnn_bbox2_adapted':
            dict(
                type='Linear',
                tensor_names=['rcnn_bbox2'],
                in_features=1024,
                out_features=1024,
            ),
        },
        losses=dict(
            cross=dict(
                type='MSELoss',
                tensor_names=['rcnn_bbox_adapted', 'teacher_rcnn_bbox0'],
                weight=2.0,
            ),
            mimic_rcnn_aux0=dict(
                type='MSELoss',
                tensor_names=['rcnn_bbox_aux0_adapted', 'teacher_rcnn_bbox_aux0'],
                weight=1.0,
            ),
            mimic_rcnn0=dict(
                type='MSELoss',
                tensor_names=['rcnn_bbox0_adapted', 'teacher_rcnn_bbox0'],
                weight=1.0,
            ),
            mimic_rcnn_aux1=dict(
                type='MSELoss',
                tensor_names=['rcnn_bbox_aux1_adapted', 'teacher_rcnn_bbox_aux1'],
                weight=1.0,
            ),
            mimic_rcnn1=dict(
                type='MSELoss',
                tensor_names=['rcnn_bbox1_adapted', 'teacher_rcnn_bbox1'],
                weight=1.0,
            ),
            mimic_rcnn_aux2=dict(
                type='MSELoss',
                tensor_names=['rcnn_bbox_aux2_adapted', 'teacher_rcnn_bbox_aux2'],
                weight=1.0,
            ),
            mimic_rcnn2=dict(
                type='MSELoss',
                tensor_names=['rcnn_bbox2_adapted', 'teacher_rcnn_bbox2'],
                weight=1.0,
            ),
        ),
        schedulers=dict(
            warmup=dict(
                type='WarmupScheduler',
                tensor_names=[
                    # 'loss_cross', 
                    'loss_mimic_rcnn0', 'loss_mimic_rcnn_aux0',
                    'loss_mimic_rcnn1', 'loss_mimic_rcnn_aux1',
                    'loss_mimic_rcnn2', 'loss_mimic_rcnn_aux2',
                ],
                iter_=2000,
            ),
        ),
    ),
)
