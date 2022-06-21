_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    'coco_detection_mstrain.py',
    'schedule_1x.py',
    '../_base_/default_runtime.py',
]
