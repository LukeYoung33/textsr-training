"""Source-line-balanced sampling for segment manifests."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class SourceLineBalancedSamplerV3(Sampler[int]):
    """Sample source lines uniformly, then cycle through their segments."""

    def __init__(self, source_line_ids: Sequence[str], seed: int = 20260815, shuffle: bool = True):
        """@shanlan-comment Group segment indices by source line and configure deterministic sampling."""
        self.groups = defaultdict(list)
        for index, line_id in enumerate(source_line_ids):
            self.groups[str(line_id)].append(index)
        if not self.groups:
            raise ValueError("source_line_ids cannot be empty")
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """@shanlan-comment Set the deterministic epoch seed."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return sum(len(values) for values in self.groups.values())

    def __iter__(self) -> Iterator[int]:
        """@shanlan-comment Emit every segment exactly once while interleaving source lines uniformly."""
        rng = np.random.default_rng(self.seed + self.epoch)
        groups = {key: list(values) for key, values in self.groups.items()}
        if self.shuffle:
            for values in groups.values():
                rng.shuffle(values)
        keys = list(groups)
        if self.shuffle:
            rng.shuffle(keys)
        emitted = 0
        while emitted < len(self):
            active = [key for key in keys if groups[key]]
            if self.shuffle:
                rng.shuffle(active)
            for key in active:
                if groups[key]:
                    yield groups[key].pop(0)
                    emitted += 1
