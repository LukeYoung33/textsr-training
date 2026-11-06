import math
import torch
from torch.utils.data.sampler import Sampler


class EnlargedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        grouped = getattr(dataset, "grouped_sampling_indices", lambda: [])()
        self.grouped_indices = grouped if grouped and sum(len(group) for group in grouped) == len(dataset) else None

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.grouped_indices is None:
            indices = torch.randperm(self.total_size, generator=g).tolist()
            dataset_size = len(self.dataset)
            indices = [v % dataset_size for v in indices]
        else:
            group_order = torch.randperm(len(self.grouped_indices), generator=g).tolist()
            indices = []
            for group_index in group_order:
                group = self.grouped_indices[group_index]
                order = torch.randperm(len(group), generator=g).tolist()
                indices.extend(group[item] for item in order)
            repeats = math.ceil(self.total_size / len(indices))
            indices = (indices * repeats)[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
