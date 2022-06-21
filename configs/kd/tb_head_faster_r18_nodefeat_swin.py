_base_ = [
    'tb_head_faster_r18_nodefeat.py',
]

model = dict(
    distiller=dict(
        teacher=dict(
            config='configs/swin/faster_rcnn_swin-t-p4-w7_fpn_fp16_1x_coco.py',
            ckpt='data/ckpts/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth',
        ),
    ),
)