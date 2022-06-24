_base_ = [
    'tb_head.py',
]
'''2x Cos 0.03.'''
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=3e-4)
runner = dict(max_epochs=24)
optimizer = dict(lr=0.03)

model = dict(
    rpn_head=dict(cache_feat_mask=True),
    distiller=dict(
        teacher=dict(
            ckpt=('data/ckpts/'
                  'mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco_'
                  '20200515_080947-8ed58c1b.pth'), ),
        student_trackings=dict(
            feat_mask=dict(type='StandardHook', path='rpn_head.feat_mask'), ),
        losses=dict(
            loss_mimic_neck=dict(
                type='DeFeatLoss',
                fields=['neck_adapted', 'teacher_neck', 'feat_mask'],
                weight=1.0,
                pos_share=0.1,
            ), ),
        schedulers=dict(early_stop=dict(iter_=7330 * 16, ), ),
    ),
)
