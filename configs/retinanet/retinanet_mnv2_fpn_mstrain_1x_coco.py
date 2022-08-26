_base_ = [
    '../_base_/models/retinanet.py',
    '../_base_/models/mnv2_fpn.py',
    '../_base_/datasets/coco_detection_mstrain.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

optimizer = dict(lr=0.01)
