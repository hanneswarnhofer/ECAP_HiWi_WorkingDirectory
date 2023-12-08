import numpy as np
import torch
from torch.utils.data import Dataset


class PytorchImageDataset(Dataset):

    def __init__(self, dset, transform=None, target_transform=None):
        # self.labels = tuple(torch.tensor(val) for k, val in dset.labels.items())
        # self.feat = tuple(torch.tensor(val) for k, val in dset.feat.items())
        self.labels = dset.labels
        self.feat = {k: np.moveaxis(val, -1, 1) for k, val in dset.feat.items()}
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels["primary"])

    def map(self, transform=None, target_transform=None):
        from torchvision.transforms import Compose
        from copy import deepcopy
        mapped_dset = deepcopy(self)

        target_transforms = [t for t in [self.target_transform, target_transform] if t is not None]
        mapped_dset.target_transform = Compose(target_transforms)

        transforms = [t for t in [self.transform, transform] if t is not None]
        mapped_dset.transform = Compose(transforms)

        return mapped_dset

    def __getitem__(self, idx):
        feat = {k: torch.tensor(val[idx], dtype=torch.float32) for k, val in self.feat.items()}
        # for k, val in self.labels.items():
        #     labels[k] = torch.tensor([val[idx]], dtype=torch.float32)[:, None]
        labels = {k: torch.tensor(val[idx], dtype=torch.float32) for k, val in self.labels.items()}
        # labels["primary"] = F.one_hot(labels["primary"].long(), 2).float()

        if self.transform:
            feat = self.transform(feat)

        if self.target_transform:
            labels = self.target_transform(labels)

        return feat, labels

    def get_loader(self, batch_size, shuffle):
        from torch.utils.data import DataLoader
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
