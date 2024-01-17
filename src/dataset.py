import os
from typing import Optional

import pandas as pd
import torch
import torchvision
from fedlab.contrib.dataset import Subset
from fedlab.utils.dataset.functional import (
    balance_split,
    client_inner_dirichlet_partition,
    hetero_dir_partition,
    shards_partition,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import even_class_split

CLASS_NUM = {
    "cifar10": 10,
    "mnist": 10,
    "fmnist": 10,
}


class PartitionedDataset:
    def __init__(
        self,
        root: str,
        path: str,
        num_clients: int,
        partition: str,
        num_shards_per_client: int,
        dir_alpha: float,
        task: str,
        public_private_split: Optional[str],
        public_size: int,
        private_size: int,
    ) -> None:
        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.partition = partition
        self.num_shards_per_client = num_shards_per_client
        self.dir_alpha = dir_alpha
        self.task = task
        self.public_private_split = public_private_split
        self.public_size = public_size
        self.private_size = private_size
        self.synthetic_dataset = None

        if self.task in ["cifar10"]:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(32, padding=4),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            self.test_transform = self.transform
        elif self.task in ["mnist", "fmnist"]:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                ]
            )
            self.test_transform = transforms.ToTensor()

        self.preprocess()

    def preprocess(self):
        """Preprocess dataset and save to local file."""

        if os.path.exists(self.path) is not True:
            os.makedirs(self.root, exist_ok=True)
            os.makedirs(self.path, exist_ok=True)
            os.mkdir(os.path.join(self.path, "private"))
            os.mkdir(os.path.join(self.path, "public"))

        match self.task:
            case "cifar10":
                self.trainset = torchvision.datasets.CIFAR10(
                    root=self.root, train=True, download=True
                )
                self.testset = torchvision.datasets.CIFAR10(
                    root=self.root,
                    train=False,
                    download=True,
                    transform=self.test_transform,
                )
            case "mnist":
                self.trainset = torchvision.datasets.MNIST(
                    root=self.root, train=True, download=True
                )
                self.testset = torchvision.datasets.MNIST(
                    root=self.root,
                    train=False,
                    download=True,
                    transform=self.test_transform,
                )
            case "fmnist":
                self.trainset = torchvision.datasets.FashionMNIST(
                    root=self.root, train=True, download=True
                )
                self.testset = torchvision.datasets.FashionMNIST(
                    root=self.root,
                    train=False,
                    download=True,
                    transform=self.test_transform,
                )
            case _:
                raise ValueError(f"Invalid dataset task: {self.task}")

        trainset_targets = self.trainset.targets

        if self.public_private_split is not None:
            assert self.public_size + self.private_size <= len(self.trainset)

            if self.public_private_split == "even_class":
                public_indices, private_indices = even_class_split(
                    dataset=self.trainset,
                    size_list=[self.public_size, self.private_size],
                )
                self.private_indices = private_indices
            elif self.public_private_split == "random_sample":
                trainset_indices = [i for i in range(len(self.trainset))]
                public_indices, private_indices, _ = torch.utils.data.random_split(
                    trainset_indices,
                    [
                        self.public_size,
                        self.private_size,
                        len(self.trainset) - self.public_size - self.private_size,
                    ],
                )
            else:
                raise ValueError(
                    f"Invalid public_private_split: {self.public_private_split}"
                )

            trainset_targets = [trainset_targets[i] for i in private_indices]
            subset_index_to_original_index = {
                i: original_index for i, original_index in enumerate(private_indices)
            }

        # get client dict
        match self.partition:
            case "shards":
                client_dict = shards_partition(
                    targets=trainset_targets,
                    num_clients=self.num_clients,
                    num_shards=self.num_clients * self.num_shards_per_client,
                )
            case "hetero_dir":
                client_dict = hetero_dir_partition(
                    targets=trainset_targets,
                    num_clients=self.num_clients,
                    num_classes=CLASS_NUM[self.task],
                    dir_alpha=self.dir_alpha,
                )
            case "client_inner_dirichlet":
                client_dict = client_inner_dirichlet_partition(
                    targets=trainset_targets,
                    num_clients=self.num_clients,
                    num_classes=CLASS_NUM[self.task],
                    dir_alpha=self.dir_alpha,
                    client_sample_nums=balance_split(
                        self.num_clients, len(trainset_targets)
                    ),
                    verbose=True,
                )
            case _:
                raise ValueError(f"Invalid partition method: {self.partition}")

        # get subsets for each client
        subsets = dict()
        self.client_dict = dict()
        for cid in range(self.num_clients):
            indices = client_dict[cid]
            if self.public_private_split is not None:
                indices = [subset_index_to_original_index[i] for i in indices]
            self.client_dict[cid] = indices
            subset = Subset(
                dataset=self.trainset, indices=indices, transform=self.transform
            )
            subsets[cid] = subset

        # save private subsets to pkl files
        for cid in subsets.keys():
            torch.save(
                subsets[cid], os.path.join(self.path, "private", f"{cid:03}.pkl")
            )

        # save public subset to pkl file
        if self.public_private_split is not None:
            public_subset = Subset(
                dataset=self.trainset, indices=public_indices, transform=self.transform
            )
            torch.save(public_subset, os.path.join(self.path, "public", "public.pkl"))

    def get_client_stats(self) -> pd.DataFrame:
        "Get statistics of the dataset for each client."

        self.stats_dict = dict()
        for cid, indices in self.client_dict.items():
            class_count = [0] * CLASS_NUM[self.task]
            for index in indices:
                class_count[self.trainset.targets[index]] += 1
            self.stats_dict[cid] = class_count
        stats_df = pd.DataFrame.from_dict(
            self.stats_dict,
            orient="index",
            columns=list(map(str, range(CLASS_NUM[self.task]))),
        )
        return stats_df

    def get_dataset(self, type, cid=None) -> Dataset:
        """Load dataset for client with client ID ``cid`` from local file.

        Args:
            type (str): Dataset type, can be ``"private"``, ``"public"`` or ``"test"``.
            cid (int, optional): client id
        """
        match type:
            case "private":
                assert cid is not None
                dataset = torch.load(
                    os.path.join(self.path, type, f"{cid:03}.pkl".format(cid))
                )
            case "public":
                dataset = torch.load(os.path.join(self.path, type, f"{type}.pkl"))
            case "test":
                dataset = self.testset
            case _:
                raise ValueError(f"Invalid dataset type: {type}")
        return dataset

    def get_dataloader(self, type: str, batch_size: int, cid=None) -> DataLoader:
        """Return dataloader for client with client ID ``cid``.

        Args:
            type (str): Dataset type, can be ``"private"``, ``"public"`` or ``"test"``.
            batch_size (int): batch size in DataLoader.
            cid (int, optional): client id, only needed when ``type`` is ``"private"``.
        """
        dataset = self.get_dataset(cid=cid, type=type)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(type != "test")
        )
        return dataloader
