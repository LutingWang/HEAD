import os

import torch
from mmcv.cnn import NORM_LAYERS
from mmcv.runner import TextLoggerHook
from todd.base import Config, DebugMode, get_logger


class Debug:
    CUSTOM = DebugMode()
    TRAIN_WITH_VAL_DATASET = DebugMode()
    LESS_DATA = DebugMode()
    LESS_BBOXES = DebugMode()
    CPU = DebugMode()
    SMALLER_BATCH_SIZE = DebugMode()
    FREQUENT_EVAL = DebugMode()


def odps_init():
    logger = get_logger()
    logger.debug("ODPS initializing.")

    def _dump_log(*args, **kwargs):
        return

    TextLoggerHook._dump_log = _dump_log
    if not os.path.lexists('data'):
        os.symlink('/data/oss_bucket_0', 'data')
    if not os.path.lexists('pretrained'):
        os.symlink('/data/oss_bucket_0/ckpts', 'pretrained')
    if not os.path.lexists('work_dirs'):
        os.symlink('/data/oss_bucket_0/work_dirs', 'work_dirs')

    logger.debug(f"ODPS initialization done with {os.listdir('.')}.")


def debug_init(debug: bool, cfg: Config):
    if torch.cuda.is_available():
        if debug:
            Debug.LESS_DATA = True
        assert not Debug.CPU
    else:
        Debug.TRAIN_WITH_VAL_DATASET = True
        Debug.LESS_DATA = True
        Debug.LESS_BBOXES = True
        Debug.CPU = True
        Debug.SMALLER_BATCH_SIZE = True
        Debug.FREQUENT_EVAL = True

    if (
        Debug.TRAIN_WITH_VAL_DATASET and 'data' in cfg and 'train' in cfg.data
        and 'val' in cfg.data
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

    if Debug.CPU:
        cfg.fp16 = None
        NORM_LAYERS.register_module(
            'SyncBN',
            force=True,
            module=NORM_LAYERS.get('BN'),
        )

    if Debug.SMALLER_BATCH_SIZE and 'data' in cfg:
        cfg.data.samples_per_gpu = 2

    if Debug.FREQUENT_EVAL and 'evaluation' in cfg:
        cfg.evaluation.interval = 1
