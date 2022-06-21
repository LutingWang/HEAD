_base_ = [
    'fgfi_r18.py',
]

model = dict(
    # dev=True,
    dev1=True,
    distiller=dict(
        dev1=True,
        teacher_online=True,
        teacher_hooks=dict(teacher_neck=None),
        # adapts=dict(neck_adapted=dict(type='Null')),
        losses=dict(feat=dict(type='MSELoss')),
    ))
find_unused_parameters = True
