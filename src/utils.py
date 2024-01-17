import argparse
import os
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def seed_everything(seed: int) -> None:
    "https://www.kaggle.com/code/rhythmcam/random-seed-everything"
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomDataset(Dataset):
    """
    Custom Dataset for PyTorch
    Args:
        data (list): list of data
        labels (list): list of labels
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """

    def __init__(self, data: list, labels: Optional[list], transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)
        if self.labels is None:
            return sample

        label = self.labels[idx]
        return sample, label


def even_class_split(dataset: Dataset, size_list: list[int]) -> list[list[int]]:
    class_indices = defaultdict(list)
    if not hasattr(dataset, "targets"):
        raise ValueError("dataset must have targets attribute")
    for i, target in enumerate(dataset.targets):
        if type(target) == torch.Tensor:
            target = target.item()
        class_indices[target].append(i)

    results = []

    for i, size in enumerate(size_list):
        clipped_indices = []

        if size % len(class_indices) != 0:
            raise ValueError(
                f"size_list[{i}] ({size}) must be divisible by the number of classes ({len(class_indices)})"
            )

        for _, indices in class_indices.items():
            np.random.shuffle(indices)
            clipped_indices.extend(indices[: size // len(class_indices)])
            indices = indices[size // len(class_indices) :]

        results.append(clipped_indices)

    return results
