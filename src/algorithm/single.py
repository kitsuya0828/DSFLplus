import os
from collections import defaultdict
from logging import Logger
from typing import DefaultDict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from algorithm.base import BaseSerialClientTrainer, BaseServerHandler
from dataset import PartitionedDataset


class SingleSerialClientTrainer(BaseSerialClientTrainer):
    """Single client with local SGD solver."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_clients: int,
        state_dict_dir: str,
        logger: Logger,
        cuda=False,
        device=None,
        personal=False,
    ) -> None:
        super().__init__(model, num_clients, cuda, device, personal)

        self.id_to_state_dict_path: DefaultDict[int, str] = defaultdict(str)
        self.state_dict_dir = state_dict_dir
        os.makedirs(self.state_dict_dir, exist_ok=True)
        self.logger = logger

    def setup_dataset(self, dataset: PartitionedDataset):
        self.dataset = dataset

    def local_process(self, payload: list, id_list: list[int], round: int):
        for id in tqdm(id_list, desc=f"Round {round}: Training", leave=False):
            data_loader = self.dataset.get_dataloader(
                type="private", batch_size=self.batch_size, cid=id
            )
            # first time training for this client
            if self.id_to_state_dict_path[id] == "":
                self.id_to_state_dict_path[id] = os.path.join(
                    self.state_dict_dir, f"{id:03}.pt"
                )

            pack = self.train(
                state_dict_path=self.id_to_state_dict_path[id],
                train_loader=data_loader,
            )
            self.cache.append(pack)

    def train(self, state_dict_path: str, train_loader: DataLoader):
        if os.path.isfile(state_dict_path):
            self.model.load_state_dict(torch.load(state_dict_path)["model_state_dict"])
            self.optimizer.load_state_dict(
                torch.load(state_dict_path)["optimizer_state_dict"]
            )
        else:
            self.setup_optim(self.epochs, self.batch_size, self.lr)
        self.model.train()

        for _ in range(self.epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            state_dict_path,
        )

        return [torch.empty(0)]


class SingleServerHandler(BaseServerHandler):
    def __init__(
        self,
        model: torch.nn.Module,
        global_round: int,
        sample_ratio: float,
        cuda: bool,
        logger: Logger,
    ):
        super().__init__(model, global_round, sample_ratio, cuda)
        self.logger = logger

    def global_update(self, buffer):
        pass
