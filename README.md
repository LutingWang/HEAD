# HEAD

[![wakatime](https://wakatime.com/badge/github/LutingWang/HEAD.svg)](https://wakatime.com/badge/github/LutingWang/HEAD)

HEtero-Assists Distillation for Heterogeneous Object Detectors

## Preparation

Download the [MS-COCO](https://cocodataset.org/#download) dataset to `data/coco`.

Download `MMDetection` pretrained models to `pretrained/mmdetection`

```bash
mkdir -p pretrained/mmdetection
wget -P pretrained/mmdetection https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth
```

Download `torchvision` pretrained models to `pretrained/torchvision`

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
ln -s ~/.cache/torch/hub/checkpoints pretrained/torchvision
wget -P pretrained/torchvision https://download.pytorch.org/models/resnet18-f37072fd.pth
wget -P pretrained/torchvision https://download.pytorch.org/models/resnet50-0676ba61.pth
```

## Installation

Create a conda environment and activate it.

```bash
conda create -n HEAD python=3.8
conda activate HEAD
```

Install `MMDetection` following [official instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation).
Specifically, the following specifications are recommended

```
torch==1.9.1
torchvision==0.10.0
mmcv_full==1.4.6
mmdet==2.20
```

Install `todd`

```bash
pip install todd_ai==0.1.6
```

## train

```bash
python tools/train.py configs/HEAD/head_retina_faster_r18_fpn_mstrain_1x_coco.py --work-dir work_dirs/debug
```

# Developer Guides

## Local Installation

```bash
pip install https://download.pytorch.org/whl/cpu/torch-1.9.1-cp38-none-macosx_11_0_arm64.whl
pip install https://download.pytorch.org/whl/cpu/torchvision-0.10.0-cp38-cp38-macosx_11_0_arm64.whl
pip install -e ./../mmcv
pip install mmdet==2.20
```

```bash
pip install -U pre-commit
pre-commit install
pre-commit install -t commit-msg
brew install commitizen
```

## TODO

- complete distributed train/test guide
- more configs
- etc.
