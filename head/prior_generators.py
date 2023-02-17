from typing import List, Protocol, Tuple, Union

import einops
import torch
from mmdet.core.anchor import AnchorGenerator as _AnchorGenerator
from mmdet.core.anchor import MlvlPointGenerator as _MlvlPointGenerator
from mmdet.core.anchor.builder import PRIOR_GENERATORS


class PriorGeneratorProto(Protocol):
    num_base_priors: List[int]

    def single_level_grid_priors(
        self,
        featmap_size: Tuple[int, int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = 'cuda',
    ) -> torch.Tensor:
        pass


class PosMixin(PriorGeneratorProto):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.with_pos = False

    def single_level_grid_priors(
        self,
        featmap_size: Tuple[int, int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = 'cuda',
    ) -> torch.Tensor:
        priors = super().single_level_grid_priors(  # type: ignore[safe-super]
            featmap_size,
            level_idx,
            dtype,
            device,
        )
        if not self.with_pos:
            return priors
        h, w = featmap_size
        a = self.num_base_priors[level_idx]
        pos = torch.zeros(
            (h, w, a, 4),
            dtype=priors.dtype,
            device=priors.device,
        )
        pos[..., 0] = level_idx
        pos[..., 1] = einops.rearrange(
            torch.arange(h, dtype=priors.dtype, device=priors.device),
            'h -> h 1 1',
        )
        pos[..., 2] = einops.rearrange(
            torch.arange(w, dtype=priors.dtype, device=priors.device),
            'w -> 1 w 1',
        )
        pos[..., 3] = einops.rearrange(
            torch.arange(a, dtype=priors.dtype, device=priors.device),
            'a -> 1 1 a',
        )
        pos = einops.rearrange(pos, 'h w a n -> (h w a) n')
        return torch.cat((priors, pos), dim=-1)


@PRIOR_GENERATORS.register_module(force=True)
class AnchorGenerator(PosMixin, _AnchorGenerator, PriorGeneratorProto):
    pass


@PRIOR_GENERATORS.register_module(force=True)
class MlvlPointGenerator(PosMixin, _MlvlPointGenerator, PriorGeneratorProto):
    pass
