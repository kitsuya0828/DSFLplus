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
        self.targets = self.labels

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
    assert hasattr(dataset, "targets")
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


def client_inner_dirichlet_partition(
    targets, num_clients, num_classes, dir_alpha, client_sample_nums, verbose=True
):
    """Non-iid Dirichlet partition.

    The method is from The method is from paper `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_.
    This function can be used by given specific sample number for all clients ``client_sample_nums``.
    It's different from :func:`hetero_dir_partition`.

    Args:
        targets (list or numpy.ndarray): Sample targets.
        num_clients (int): Number of clients for partition.
        num_classes (int): Number of classes in samples.
        dir_alpha (float): Parameter alpha for Dirichlet distribution.
        client_sample_nums (numpy.ndarray): A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    class_priors = np.random.dirichlet(
        alpha=[dir_alpha] * num_classes, size=num_clients
    )
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]

    client_indices = [
        np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in range(num_clients)
    ]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        # If current node is full resample a client
        if verbose:
            print("Remaining Data: %d" % np.sum(client_sample_nums))
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        failed_count = 0
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if no rest in current class samples
            if class_amount[curr_class] <= 0:
                failed_count += 1
                # If failed too many times, resample a client
                if failed_count > 10**5:
                    break
                continue
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = idx_list[
                curr_class
            ][class_amount[curr_class]]

            break

    client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
    return client_dict
