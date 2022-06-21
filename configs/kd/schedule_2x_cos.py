_base_ = [
    '../_base_/schedules/schedule_2x.py',
]

optimizer = dict(lr=0.03)
optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=20),
)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=3e-4)
