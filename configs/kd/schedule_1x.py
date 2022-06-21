_base_ = [
    '../_base_/schedules/schedule_1x.py',
]

optimizer = dict(lr=0.01)
optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=20),
)
