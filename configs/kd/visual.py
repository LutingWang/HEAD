_base_ = [
    'tb_head.py',
]

model = dict(
    visualize=True,
    distiller=dict(
        student_trackings=dict(
            images=dict(type='DuplicatedHook', path='imgs', num=5), ),
        visuals=dict(
            act=dict(
                type='ActivationVisual',
                fields=['images', 'neck'],
                parallel=True,
                log_dir='work_dirs/visual/',
            ), ),
    ),
)
