_base_ = [
    'coco_detection.py',
]

data = dict(
    train=dict(
        pipeline={
            2: dict(
                img_scale=[(1333, 640), (1333, 800)],
                multiscale_mode='range',
            ),
        },
    ),
)
