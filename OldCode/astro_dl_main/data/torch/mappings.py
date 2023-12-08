import torch
import torch.nn as nn
from typing import List, Optional, Union
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data.data import BaseData
from torch_geometric.data import Dataset


class DictCollater(Collater):
    def __call__(self, batch):
        return super().__call__(batch).to_dict()


class DictDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, List[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=DictCollater(follow_batch, exclude_keys),
            **kwargs,
        )


class RmUnusedLabels(nn.Module):
    """Remove labels from the Dataset."""

    def __init__(self, exclude_keys):
        super().__init__()
        self.exclude_keys = exclude_keys

    def __call__(self, labels):
        new_labels = {}

        for k, lab_val in labels.items():

            if k not in self.exclude_keys:
                new_labels[k] = lab_val

        return new_labels
