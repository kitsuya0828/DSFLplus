import os

import torch
from fedlab.contrib.algorithm import SGDSerialClientTrainer, SyncServerHandler
from torch.utils.data import DataLoader


class BaseSerialClientTrainer(SGDSerialClientTrainer):
    def setup_datetime(self, datetime: str):
        self.datetime = datetime

    def evaluate(
        self, state_dict_path: str, test_loader: DataLoader
    ) -> tuple[float, float]:
        """Evaluate the local model on test dataset.

        Args:
            state_dict_path (str): Path to the local model state dict.
            test_loader (torch.utils.data.DataLoader): Test dataset loader.
        """
        if os.path.isfile(state_dict_path):
            self.model.load_state_dict(torch.load(state_dict_path)["model_state_dict"])
        else:
            raise ValueError(f"File {state_dict_path} not found.")
        self.model.eval()

        loss_sum = 0.0
        acc_sum = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                if self.cuda:
                    inputs = inputs.cuda(self.device)
                    labels = labels.cuda(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                loss_sum += loss.item()
                acc_sum += torch.sum(predicted.eq(labels)).item()

        assert hasattr(test_loader.dataset, "__len__")
        return loss_sum, acc_sum / len(test_loader.dataset)


class BaseServerHandler(SyncServerHandler):
    def evaluate(self, test_loader: DataLoader) -> tuple[float, float]:
        """Evaluate the global model on test dataset.

        Args:
            test_loader (torch.utils.data.DataLoader): Test dataset loader.
        """
        self.model.eval()

        loss_sum = 0.0
        acc_sum = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                if self.cuda:
                    inputs = inputs.cuda(self.device)
                    labels = labels.cuda(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                loss_sum += loss.item()
                acc_sum += torch.sum(predicted.eq(labels)).item()

        assert hasattr(test_loader.dataset, "__len__")
        return loss_sum, acc_sum / len(test_loader.dataset)
