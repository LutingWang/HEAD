# HEtero-Assists Distillation (ECCV 2022)

```text
    _/    _/  _/_/_/_/    _/_/    _/_/_/
   _/    _/  _/        _/    _/  _/    _/
  _/_/_/_/  _/_/_/    _/_/_/_/  _/    _/
 _/    _/  _/        _/    _/  _/    _/
_/    _/  _/_/_/_/  _/    _/  _/_/_/
```

This repository is the official implementation of "[HEtero-Assists Distillation for Heterogeneous Object Detectors](https://arxiv.org/abs/2207.05345)".

[![lint](https://github.com/LutingWang/HEAD/actions/workflows/lint.yaml/badge.svg)](https://github.com/LutingWang/HEAD/actions/workflows/lint.yaml)
[![wakatime](https://wakatime.com/badge/github/LutingWang/HEAD.svg)](https://wakatime.com/badge/github/LutingWang/HEAD)

## Preparation

Download the [MS-COCO](https://cocodataset.org/#download) dataset to `data/coco`.

Download `MMDetection` pretrained models to `pretrained/mmdetection`

```bash
mkdir -p pretrained/mmdetection
wget -P pretrained/mmdetection https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth
wget -P pretrained/mmdetection https://download.openmmlab.com/mmdetection/v2.0/third_party/mobilenet_v2_batch256_imagenet-ff34753d.pth
wget -P pretrained/mmdetection https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_mstrain_3x_coco/retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pth
```

Download `torchvision` pretrained models to `pretrained/torchvision`

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
ln -s ~/.cache/torch/hub/checkpoints pretrained/torchvision
wget -P pretrained/torchvision https://download.pytorch.org/models/resnet18-f37072fd.pth
wget -P pretrained/torchvision https://download.pytorch.org/models/resnet50-0676ba61.pth
```

The directory tree should be like this

```text
HEAD
├── data
│   └── coco -> ~/Developer/datasets/coco
│       ├── annotations
│       │   ├── instances_train2017.json
│       │   └── instances_val2017.json
│       ├── train2017
│       │   └── ...
│       └── val2017
│           └── ...
├── pretrained
│   ├── mmdetection
│   │   ├── faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth
│   │   ├── mobilenet_v2_batch256_imagenet-ff34753d.pth
│   │   └── retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pth
│   └── torchvision -> ~/.cache/torch/hub/checkpoints
│       ├── resnet18-f37072fd.pth
│       └── resnet50-0676ba61.pth
└── ...
```

## Installation

Create a conda environment and activate it.

```bash
conda create -n HEAD python=3.8
conda activate HEAD
```

Install `MMDetection` following the [official instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation).
For example,

```bash
pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install mmcv_full==1.4.6
pip install mmdet==2.20
```

Install `todd`.

```bash
pip install todd_ai==0.2.3a9 -i https://pypi.org/simple
```

> Note that the `requirements.txt` is not intended for users. Please follow the above instructions.

## train

```bash
python tools/train.py configs/head/head_retina_faster_r18_fpn_mstrain_1x_coco.py --work-dir work_dirs/debug --seed 3407
```

For distributed training

```bash
bash tools/dist_train.sh configs/head/head_retina_faster_r18_fpn_mstrain_1x_coco.py 8 --work-dir work_dirs/debug --seed 3407
```

## Results

All logs and checkpoints can be found in the [Google Drive](https://drive.google.com/drive/folders/1cs9WWyBaZmstsKlwnMv7PE9ky-i98WUh?usp=sharing).

### Baseline

| Method    | Student       | Teacher       | mAP       | Config                                                                    | Comment               |
| :-:       | :-:           | :-:           | :-:       | :-:                                                                       | :-:                   |
| FitNet    | R18 RetinaNet | R50 RetinaNet | $35.6$    | [fitnet_retina](configs/fitnet/fitnet_retina_r18_fpn_mstrain_1x_coco.py)  | with weight transfer  |

### HEAD

Teachers and students are all trained with multi-scale, for 3x and 1x scheduler respectively.

| Student           | Teacher           | Assist        | AKD           | CKD           | mAP       | Config                                                                                                    |
| :-:               | :-:               | :-:           | :-:           | :-:           | :-:       | -                                                                                                         |
| R18 RetinaNet     |                   |               |               |               | $31.7$    | refer to [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet/README.md) |
| R18 RetinaNet     | R50 Faster R-CNN  | $\checkmark$  |               |               | $33.4$    | [retina_faster_r18](configs/assist/retina_faster_r18_fpn_mstrain_1x_coco.py)                              |
| R18 RetinaNet     | R50 Faster R-CNN  | $\checkmark$  | $\checkmark$  |               | $35.7$    | [HEAD_dag_retina_faster_r18](configs/head_dag/head_dag_retina_faster_r18_fpn_mstrain_1x_coco.py)          |
| R18 RetinaNet     | R50 Faster R-CNN  | $\checkmark$  | $\checkmark$  | $\checkmark$  | $36.1$    | [HEAD_retina_faster_r18](configs/head/head_retina_faster_r18_fpn_mstrain_1x_coco.py)                      |
| MNv2 RetinaNet    |                   |               |               |               | $27.8$    | [retinanet_mnv2](configs/retinanet/retinanet_mnv2_fpn_mstrain_1x_coco.py)                                 |
| MNv2 RetinaNet    | R50 Faster R-CNN  | $\checkmark$  |               |               | $28.9$    | [retina_faster_mnv2](configs/assist/retina_faster_mnv2_fpn_mstrain_1x_coco.py)                            |
| MNv2 RetinaNet    | R50 Faster R-CNN  | $\checkmark$  | $\checkmark$  |               | $32.2$    | [HEAD_dag_retina_faster_mnv2](configs/head_dag/head_dag_retina_faster_mnv2_fpn_mstrain_1x_coco.py)        |
| MNv2 RetinaNet    | R50 Faster R-CNN  | $\checkmark$  | $\checkmark$  | $\checkmark$  | $33.1$    | [HEAD_retina_faster_mnv2](configs/head/head_retina_faster_mnv2_fpn_mstrain_1x_coco.py)                    |

### TF-HEAD

Coming soon...

## Developer Guides

### Setup

```bash
pip install https://download.pytorch.org/whl/cpu/torch-1.9.1-cp38-none-macosx_11_0_arm64.whl
pip install https://download.pytorch.org/whl/cpu/torchvision-0.10.0-cp38-cp38-macosx_11_0_arm64.whl
pip install -e ./../mmcv
pip install mmdet==2.20
```

```bash
pip install commitizen
pip install -U pre-commit
pre-commit install
pre-commit install -t commit-msg
```

### TODO

- complete distributed train/test guide
- more configs
- etc.
