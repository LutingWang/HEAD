from typing import Any, Dict, List, NoReturn, Optional, Tuple

import todd
import torch



@todd.adapts.ADAPTS.register_module()
class CustomAdapt(todd.adapts.BaseAdapt):

    def __init__(self, *args, stride: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._stride = stride

    def forward(
        self,
        feat: torch.Tensor,
        pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_idx = pos[:, 1] >= 0
        feat = feat[valid_idx]
        pos = pos[valid_idx]
        bs, level, h, w, id_ = pos.split(1, 1)
        h = h // self._stride
        w = w // self._stride
        pos = torch.cat((level, bs, h, w), dim=-1)
        id_ = id_.reshape(-1)
        return feat, pos, id_
