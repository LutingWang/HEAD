from typing import Any, List

from mmdet.datasets import DATASETS, CustomDataset, CocoDataset as _CocoDataset
import torch

from head.utils import has_debug_flag


class DebugMixin(CustomDataset):

    def __len__(self) -> int:
        if has_debug_flag(2):
            return 4
        return super().__len__()

    def load_annotations(self, *args, **kwargs):
        data_infos = super().load_annotations(*args, **kwargs)
        if has_debug_flag(2):
            data_infos = data_infos[:len(self)]
        return data_infos

    def load_proposals(self, *args, **kwargs):
        proposals = super().load_proposals(*args, **kwargs)
        if has_debug_flag(2):
            proposals = proposals[:len(self)]
        return proposals

    def evaluate(self, *args, **kwargs):
        kwargs.pop('gpu_collect', None)
        kwargs.pop('tmpdir', None)
        return super().evaluate(*args, **kwargs)


# @DATASETS.register_module(force=True)
class CocoDataset(DebugMixin, _CocoDataset):

    def load_annotations(self, *args, **kwargs):
        data_infos = super().load_annotations(*args, **kwargs)
        if has_debug_flag(2):
            self.coco.dataset['images'] = self.coco.dataset['images'][:len(self)]
            self.img_ids = [img['id'] for img in self.coco.dataset['images']]
            self.coco.dataset['annotations'] = [
                ann for ann in self.coco.dataset['annotations']
                if ann['image_id'] in self.img_ids
            ]
            self.coco.imgs = {img['id']: img for img in self.coco.dataset['images']}
        return data_infos
