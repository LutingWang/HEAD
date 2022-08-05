import enum
import os
import os.path as osp

import torch
from mmcv import Config
from mmcv.cnn import NORM_LAYERS
from mmcv.runner import TextLoggerHook
from todd.base import get_logger


class _DebugEnum(enum.Enum):

    @classmethod
    def is_active(cls) -> bool:
        return any(e.is_on for e in cls)

    @property
    def is_on(self) -> bool:
        return any(map(os.getenv, ['DEBUG', 'DEBUG_' + self.name]))

    def turn_on(self) -> None:
        os.environ['DEBUG_' + self.name] = '1'

    def turn_off(self) -> None:
        os.environ['DEBUG_' + self.name] = ''


class DebugEnum(_DebugEnum):
    CUSTOM = enum.auto()
    TRAIN_WITH_VAL_DATASET = enum.auto()
    LESS_DATA = enum.auto()
    LESS_BBOXES = enum.auto()
    CPU = enum.auto()
    SMALLER_BATCH_SIZE = enum.auto()
    FREQUENT_EVAL = enum.auto()


def odps_init():
    logger = get_logger()
    logger.debug("ODPS initializing.")

    def _dump_log(*args, **kwargs):
        return

    TextLoggerHook._dump_log = _dump_log
    if not osp.lexists('data'):
        os.symlink('/data/oss_bucket_0', 'data')
    if not osp.lexists('pretrained'):
        os.symlink('/data/oss_bucket_0/ckpts', 'pretrained')
    if not osp.lexists('work_dirs'):
        os.symlink('/data/oss_bucket_0/work_dirs', 'work_dirs')

    logger.debug(f"ODPS initialization done with {os.listdir('.')}.")


def debug_init(debug: bool, cfg: Config):
    if torch.cuda.is_available():
        if debug:
            DebugEnum.LESS_DATA.turn_on()
        if DebugEnum.is_active():
            assert not DebugEnum.CPU.is_on
        else:
            return
    else:
        DebugEnum.TRAIN_WITH_VAL_DATASET.turn_on()
        DebugEnum.LESS_DATA.turn_on()
        DebugEnum.LESS_BBOXES.turn_on()
        DebugEnum.CPU.turn_on()
        DebugEnum.SMALLER_BATCH_SIZE.turn_on()
        DebugEnum.FREQUENT_EVAL.turn_on()

    if (
        DebugEnum.TRAIN_WITH_VAL_DATASET.is_on and 'data' in cfg
        and 'train' in cfg.data and 'val' in cfg.data
    ):
        data_train = cfg.data.train
        data_val = cfg.data.val
        if 'dataset' in data_train:
            data_train = data_train.dataset
        if 'dataset' in data_val:
            data_val = data_val.dataset
        data_val = {  # yapf: disable
            k: v for k, v in data_val.items()
            if k in ['ann_file', 'img_prefix', 'proposal_file']
        }
        data_train.update(data_val)

    if DebugEnum.CPU.is_on:
        cfg.fp16 = None
        NORM_LAYERS.register_module(
            'SyncBN',
            force=True,
            module=NORM_LAYERS.get('BN'),
        )

    if DebugEnum.SMALLER_BATCH_SIZE.is_on and 'data' in cfg:
        cfg.data.samples_per_gpu = 2

    if DebugEnum.FREQUENT_EVAL.is_on and 'evaluation' in cfg:
        cfg.evaluation.interval = 1
