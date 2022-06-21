_base_ = [
    'retina_reppoints_mnv2.py',
]

model = dict(
    type='TB_HEADSingleStage',
    warmup=dict(
        warmup=dict(
            type='WarmupScheduler',
            tensor_names=[
                'loss_cls_reppoints',
                'loss_pts_init_reppoints',
                'loss_pts_refine_reppoints',
            ],
            iter_=2000,
            value=1.0,
        )),
    distiller=dict(
        teacher=dict(
            config='configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py',
            ckpt=('data/ckpts/reppoints_moment_r50_fpn_gn-neck+head_2x_'
                  'coco_20200329-91babaa2.pth'),
        ),
        weight_transfer={
            'student.bbox_head._extra_heads["reppoints"]': 'teacher.bbox_head',
        },
        student_hooks=dict(
            retina_cls=dict(
                type='MultiCallsHook',
                path='bbox_head.cls_convs[1]',
            ),
            retina_reg=dict(
                type='MultiCallsHook',
                path='bbox_head.reg_convs[1]',
            ),
            rp_cls=dict(
                type='MultiCallsHook',
                path='bbox_head._extra_heads["reppoints"].cls_convs[2]',
            ),
            rp_reg=dict(
                type='MultiCallsHook',
                path='bbox_head._extra_heads["reppoints"].reg_convs[2]',
            ),
            rp_pts_init_conv=dict(
                type='MultiCallsHook',
                path='bbox_head._extra_heads["reppoints"].reppoints_pts_init_conv',
            ),
            rp_cls_conv=dict(
                type='MultiCallsHook',
                path='bbox_head._extra_heads["reppoints"].reppoints_cls_conv',
            ),
            rp_pts_refine_conv=dict(
                type='MultiCallsHook',
                path='bbox_head._extra_heads["reppoints"].reppoints_pts_refine_conv',
            ),
        ),
        teacher_hooks=dict(
            teacher_rp_cls=dict(
                type='MultiCallsHook',
                path='bbox_head.cls_convs[2]',
            ),
            teacher_rp_reg=dict(
                type='MultiCallsHook',
                path='bbox_head.reg_convs[2]',
            ),
            teacher_rp_pts_init_conv=dict(
                type='MultiCallsHook',
                path='bbox_head.reppoints_pts_init_conv',
            ),
            teacher_rp_cls_conv=dict(
                type='MultiCallsHook',
                path='bbox_head.reppoints_cls_conv',
            ),
            teacher_rp_pts_refine_conv=dict(
                type='MultiCallsHook',
                path='bbox_head.reppoints_pts_refine_conv',
            ),
        ),
        adapts={
            t: dict(
                type='Conv2d',
                tensor_names=[t],
                multilevel=True,
                in_channels=256, 
                out_channels=256,
                kernel_size=1,
            ) for t in [
                'retina_cls', 'retina_reg', 
                'rp_cls', 'rp_reg', 
                'rp_pts_init_conv', 'rp_cls_conv', 'rp_pts_refine_conv',
            ]
        },
        losses=dict(
            cross_cls=dict(
                type='MSELoss',
                tensor_names=['retina_cls', 'teacher_rp_cls'],
                multilevel=True,
                weight=1.0
            ),
            cross_reg=dict(
                type='MSELoss',
                tensor_names=['retina_reg', 'teacher_rp_reg'],
                multilevel=True,
                weight=1.0
            ),
            **{
                t: dict(
                    type='MSELoss',
                    tensor_names=[t, 'teacher_' + t],
                    multilevel=True,
                    weight=1.0,
                ) for t in [
                    'rp_cls', 'rp_reg', 
                    'rp_pts_init_conv', 'rp_cls_conv', 'rp_pts_refine_conv',
                ]
            },
        ),
        schedulers=dict(
            warmup=dict(
                type='WarmupScheduler',
                tensor_names=[
                    'loss_cross_cls', 'loss_cross_reg',
                    'loss_rp_cls', 'loss_rp_reg', 
                    'loss_rp_pts_init_conv', 'loss_rp_cls_conv', 
                    'loss_rp_pts_refine_conv',
                ],
                iter_=2000,
            ),
        ),
    ),
)
