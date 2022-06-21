# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import warnings

from mmcv import Config
from mmcv.runner import TextLoggerHook
from mmcv.cnn import NORM_LAYERS
from todd.base import get_logger
import torch


def find_latest_checkpoint(path, suffix='pth'):
    """Find the latest checkpoint from the working directory.

    Args:
        path(str): The path to find checkpoints.
        suffix(str): File extension.
            Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.
    References:
        .. [1] https://github.com/microsoft/SoftTeacher
                  /blob/main/ssod/utils/patch.py
    """
    if not osp.exists(path):
        warnings.warn('The path of checkpoints does not exist.')
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    checkpoints = glob.glob(osp.join(path, f'*.{suffix}'))
    if len(checkpoints) == 0:
        warnings.warn('There are no checkpoints in the path.')
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path


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
    if torch.cuda.is_available() and not debug:
        return
    if 'DEBUG' not in os.environ:
        os.environ['DEBUG'] = '001' if torch.cuda.is_available() else '0111111'
    os.environ['DEBUG'] += '0' * 10
    if has_debug_flag(1) and 'data' in cfg and 'train' in cfg.data:
        data_train = cfg.data.train
        if 'dataset' in data_train:
            data_train = data_train.dataset
        if 'ann_file' in cfg.data.val:
            data_train.ann_file = cfg.data.val.ann_file
        if 'img_prefix' in cfg.data.val:
            data_train.img_prefix = cfg.data.val.img_prefix
        if 'proposal_file' in cfg.data.val:
            data_train.proposal_file = cfg.data.val.proposal_file
    if has_debug_flag(4):
        NORM_LAYERS.register_module('SyncBN', force=True, module=NORM_LAYERS.get('BN'))
        cfg.fp16 = None
    if has_debug_flag(5):
        cfg.data.samples_per_gpu = 2
    if has_debug_flag(6):
        cfg.evaluation.interval = 1


def has_debug_flag(level: int) -> bool:
    """Parse debug flags.

    Levels:
        0: custom flag
        1: use val dataset as train dataset
        2: use smaller datasets
        3: use fewer gt bboxes
        4: cpu
        5: use batch size 2
        6: eval interval 1

    Note:
        Flag 1/2/3 are set by default when cuda is unavailable.
    """
    if 'DEBUG' not in os.environ:
        return False
    return bool(int(os.environ['DEBUG'][level]))
