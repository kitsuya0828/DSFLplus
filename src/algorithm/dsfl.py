import os
from collections import defaultdict
from logging import Logger
from typing import DefaultDict, List, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from algorithm.base import BaseSerialClientTrainer, BaseServerHandler
from dataset import PartitionedDataset
from utils import CustomDataset


class DSFLSerialClientTrainer(BaseSerialClientTrainer):
    """DSFL client with local SGD solver."""

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
        self.public_dataset = dataset.get_dataset(type="public")

    def setup_kd_optim(self, epochs: int, batch_size: int, lr: float):
        self.kd_epochs = epochs
        self.kd_batch_size = batch_size
        self.kd_lr = lr
        self.kd_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.kd_lr)
        self.kd_criterion = torch.nn.KLDivLoss(reduction="batchmean")

    def local_process(self, payload: list, id_list: list[int], round: int):
        global_logits = payload[0]
        global_indices = payload[1]
        next_indices = payload[2]
        self.round = round
        for id in tqdm(id_list, desc=f"Round {round}: Training", leave=False):
            self.current_client_id = id
            data_loader = self.dataset.get_dataloader(
                type="private", batch_size=self.batch_size, cid=id
            )
            # first time training for this client
            if self.id_to_state_dict_path[id] == "":
                self.id_to_state_dict_path[id] = os.path.join(
                    self.state_dict_dir, f"{id:03}.pt"
                )

            self.train(
                state_dict_path=self.id_to_state_dict_path[id],
                global_logits=global_logits,
                global_indices=global_indices,
                train_loader=data_loader,
            )
            pack = self.predict(next_indices)
            self.cache.append(pack)

    def train(
        self, state_dict_path, global_logits, global_indices, train_loader
    ) -> None:
        """Train model with local dataset.

        Args:
            state_dict_path (str): path to load and save state dict.
            global_logits (torch.Tensor): global logits received from server.
            global_indices (torch.Tensor): global indices received from server.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        if os.path.isfile(state_dict_path):
            self.model.load_state_dict(torch.load(state_dict_path)["model_state_dict"])
            self.optimizer.load_state_dict(
                torch.load(state_dict_path)["optimizer_state_dict"]
            )
            self.kd_optimizer.load_state_dict(
                torch.load(state_dict_path)["kd_optimizer_state_dict"]
            )
        else:
            self.setup_optim(self.epochs, self.batch_size, self.lr)
            self.setup_kd_optim(self.kd_epochs, self.kd_batch_size, self.kd_lr)
        self.model.train()

        if global_logits is not None:
            public_subset = Subset(self.public_dataset, global_indices)
            public_loader = DataLoader(public_subset, batch_size=self.batch_size)
            public_logits_loader = DataLoader(
                CustomDataset(data=torch.unbind(global_logits, dim=0), labels=None),
                batch_size=self.kd_batch_size,
            )
            for _ in range(self.kd_epochs):
                for batch_idx, ((data, _), logit) in enumerate(
                    zip(public_loader, public_logits_loader)
                ):
                    if self.cuda:
                        data = data.cuda(self.device)
                        logit = logit.cuda(self.device)

                    output = F.log_softmax(self.model(data), dim=1)
                    logit = logit.squeeze(1)
                    kd_loss = self.kd_criterion(output, logit)

                    self.kd_optimizer.zero_grad()
                    kd_loss.backward()
                    self.kd_optimizer.step()

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
                "kd_optimizer_state_dict": self.kd_optimizer.state_dict(),
            },
            state_dict_path,
        )

    def predict(self, public_indices: torch.Tensor) -> List[torch.Tensor]:
        """Predict for public dataset.

        Args:
            public_indices (torch.Tensor): indices of public dataset to predict.
        """
        self.model.eval()

        tmp_local_logits: List[torch.Tensor] = []
        with torch.no_grad():
            predict_subset = Subset(self.public_dataset, public_indices.tolist())
            predict_loader = DataLoader(
                predict_subset, batch_size=min(self.batch_size, len(public_indices))
            )
            for data, _ in predict_loader:
                if self.cuda:
                    data = data.cuda(self.device)

                output = self.model(data)
                logits = F.softmax(output, dim=1)
                tmp_local_logits.extend([logit.detach().cpu() for logit in logits])

        local_logits = torch.stack(tmp_local_logits)
        local_indices = torch.tensor(public_indices.tolist())

        return [local_logits, local_indices]


class DSFLServerHandler(BaseServerHandler):
    "DSFL server handler."

    def __init__(
        self,
        model: torch.nn.Module,
        global_round: int,
        sample_ratio: float,
        cuda: bool,
        temperature: float,
        public_size_per_round: int,
        logger,
    ):
        super().__init__(model, global_round, sample_ratio, cuda)
        self.global_logits: Union[torch.Tensor, None] = None
        self.global_indices: Union[torch.Tensor, None] = None
        self.temperature = temperature
        self.public_size_per_round = public_size_per_round
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = logger

    def setup_kd_optim(self, kd_epochs: int, kd_batch_size: int, kd_lr: float):
        """Setup optimizer for knowledge distillation.

        Args:
            kd_epochs (int): epochs for knowledge distillation.
            kd_batch_size (int): batch size for knowledge distillation.
            kd_lr (float): learning rate for knowledge distillation.
        """
        self.kd_epochs = kd_epochs
        self.kd_batch_size = kd_batch_size
        self.kd_lr = kd_lr
        self.kd_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.kd_lr)
        self.kd_criterion = torch.nn.KLDivLoss(reduction="batchmean")

    def setup_dataset(self, dataset: PartitionedDataset) -> None:
        """Setup dataset for server handler.

        Args:
            dataset (PartitionedDataset): partitioned dataset.
        """
        self.public_dataset = dataset.get_dataset(type="public")
        self.set_next_public_indices(size=self.public_size_per_round)

    def set_next_public_indices(self, size: int) -> None:
        """Set public indices for next round.

        Args:
            size (int): size of next public indices.
        """
        assert hasattr(self.public_dataset, "__len__")

        size = min(size, len(self.public_dataset))

        shuffled_indices = torch.randperm(len(self.public_dataset))
        self.global_next_indices = shuffled_indices[:size]

    def global_update(self, buffer: list) -> None:
        "Update global model with local parameters."
        logits_list = [ele[0] for ele in buffer]
        indices_list = [ele[1] for ele in buffer]

        global_logits_stack = defaultdict(list)
        for logits, indices in zip(logits_list, indices_list):
            for logit, indice in zip(logits, indices):
                global_logits_stack[indice.item()].append(logit)

        global_logits: List[torch.Tensor] = []
        global_indices: List[int] = []
        for indice, logits in global_logits_stack.items():
            global_indices.append(indice)
            # Entropy Reduction Aggregation
            mean_logit = torch.stack(logits).mean(dim=0).cpu()
            era_logit = F.softmax(mean_logit / self.temperature, dim=0)
            global_logits.append(era_logit)

        # update global model
        self.model.train()
        global_subset = Subset(self.public_dataset, global_indices)
        global_loader = DataLoader(global_subset, batch_size=self.kd_batch_size)
        global_logits_loader = DataLoader(
            CustomDataset(data=global_logits, labels=None),
            batch_size=self.kd_batch_size,
        )
        for _ in range(self.kd_epochs):
            for batch_idx, ((data, target), logit) in enumerate(
                zip(global_loader, global_logits_loader)
            ):
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)
                    logit = logit.cuda(self.device)

                output = F.log_softmax(self.model(data), dim=1)
                logit = logit.squeeze(1)
                kd_loss = self.kd_criterion(output, logit)

                self.kd_optimizer.zero_grad()
                kd_loss.backward()
                self.kd_optimizer.step()

        # prepare package
        self.global_indices = torch.tensor(global_indices)
        self.global_logits = torch.stack(global_logits)

        self.set_next_public_indices(size=self.public_size_per_round)

    @property
    def downlink_package(self) -> List[Union[torch.Tensor, None]]:
        return [self.global_logits, self.global_indices, self.global_next_indices]
