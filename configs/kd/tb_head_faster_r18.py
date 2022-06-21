_base_ = [
    'head_faster_r18.py',
]

model = dict(
    type='TB_HEAD',
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
            rcnn_bbox_aux=dict(
                type='StandardHook',
                path='roi_head.bbox_head.shared_fcs[0]',
            ),
        ),
        teacher_hooks=dict(
            teacher_neck=dict(type='StandardHook', path='neck'),
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
            'neck_adapted':
            dict(
                type='Conv2d',
                tensor_names=['neck'],
                multilevel=True,
                in_channels=256,
                out_channels=256,
                kernel_size=1,
            ),
            'masks':
            dict(
                type='DeFeatMask',
                tensor_names=['batch_input_shape', 'gt_bboxes'],
                neg_gain=9,
                strides=[8, 16, 32, 64, 128],
                ceil_mode=True,
            ),
            'rcnn_bbox_aux_adapted':
            dict(
                type='Linear',
                tensor_names=['rcnn_bbox_aux'],
                in_features=1024,
                out_features=1024,
            ),
            'rcnn_bbox_adapted':
            dict(
                type='Linear',
                tensor_names=['rcnn_bbox'],
                in_features=1024,
                out_features=1024,
            ),
        },
        losses=dict(
            mimic_neck=dict(
                type='MSELoss',
                tensor_names=['neck_adapted', 'teacher_neck', 'masks'],
                multilevel=True,
                weight=1.0,
            ),
            mimic_rcnn_aux=dict(
                type='MSELoss',
                tensor_names=['rcnn_bbox_aux_adapted', 'teacher_rcnn_bbox_aux'],
                weight=2.0,
            ),
            mimic_rcnn=dict(
                type='MSELoss',
                tensor_names=['rcnn_bbox_adapted', 'teacher_rcnn_bbox'],
                weight=2.0,
            ),
        ),
        schedulers=dict(
            warmup=dict(tensor_names=[
                'loss_self_rcnn', 'loss_mimic_neck',
                'loss_mimic_rcnn', 'loss_mimic_rcnn_aux',
            ]),
        ),
    ),
)
