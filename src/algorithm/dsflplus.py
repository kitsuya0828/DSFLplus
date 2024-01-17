import os
from logging import Logger
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from algorithm.dsfl import DSFLSerialClientTrainer, DSFLServerHandler


class DSFLPlusSerialClientTrainer(DSFLSerialClientTrainer):
    """DSFL+ client with local SGD solver."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_clients: int,
        state_dict_dir: str,
        logger: Logger,
        ood_detection_type: str,
        ood_detection_schedule: str,
        cuda=False,
        device=None,
        personal=False,
    ) -> None:
        super().__init__(
            model, num_clients, state_dict_dir, logger, cuda, device, personal
        )
        self.ood_detection_type = ood_detection_type
        if self.ood_detection_type is not None and self.ood_detection_type.endswith(
            "schedule"
        ):
            self.ood_detection_schedule_list = self._get_schedule_list(
                ood_detection_schedule, end=1000
            )

    def _get_schedule_list(self, schedule: str, end: int) -> list:
        """Get schedule list from schedule string."""
        match schedule:
            case "linear_25_100":
                arr = np.linspace(0.25, 1.0, end)
            case "linear_50_100":
                arr = np.linspace(0.5, 1.0, end)
            case "linear_75_100":
                arr = np.linspace(0.75, 1.0, end)
            case _:
                raise NotImplementedError
        return arr.tolist()

    def predict(self, public_indices: torch.Tensor) -> List[torch.Tensor]:
        """Predict for public dataset.

        Args:
            public_indices (torch.Tensor): indices of public dataset to predict.
        """
        self.model.eval()

        tmp_local_logits: List[torch.Tensor] = []
        local_outputs = []  # for ood detection
        local_targets = []  # for ood detection
        with torch.no_grad():
            predict_subset = Subset(self.public_dataset, public_indices.tolist())
            predict_loader = DataLoader(
                predict_subset, batch_size=min(self.batch_size, len(public_indices))
            )
            for data, target in predict_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    # target is not used for prediction, but for ood detection analysis
                    target = target.cuda(self.device)

                output = self.model(data)

                local_outputs.extend([o.detach() for o in output])
                local_targets.extend([t.detach() for t in target])

                logits = F.softmax(output, dim=1)
                tmp_local_logits.extend([logit.detach().cpu() for logit in logits])

            local_logits = torch.stack(tmp_local_logits)
            local_indices = torch.tensor(public_indices.tolist())

        if self.ood_detection_type is not None:
            if self.ood_detection_type.startswith("energy"):
                negative_energy = torch.log(
                    torch.exp(torch.stack(local_outputs)).sum(dim=1)
                )

                if self.ood_detection_type == "energy_mean":
                    threshold = negative_energy.mean()
                elif self.ood_detection_type == "energy_median":
                    threshold = negative_energy.median()
                elif (
                    self.ood_detection_type == "energy_75percentile"
                ):  # negative energy 25% percentile
                    threshold = negative_energy.quantile(0.25)
                elif (
                    self.ood_detection_type == "energy_25percentile"
                ):  # negative energy 75% percentile
                    threshold = negative_energy.quantile(0.75)
                elif self.ood_detection_type == "energy_schedule":
                    if self.round < len(self.ood_detection_schedule_list):
                        q = self.ood_detection_schedule_list[self.round]  # energy
                        threshold = negative_energy.quantile(1 - q)  # negative energy
                    else:  # energy max
                        threshold = negative_energy.min()
                else:
                    raise NotImplementedError

                # classify input as in-distribution if negative energy is larger than the threshold value
                id_indices = (negative_energy >= threshold).nonzero().squeeze(1).cpu()

                if self.current_client_id == 0:
                    os.makedirs(f"tmp/{self.datetime}", exist_ok=True)
                    np.savez(
                        f"tmp/{self.datetime}/{self.round}.npz",
                        negative_energy=negative_energy.cpu().numpy(),
                        target=torch.stack(local_targets).cpu().numpy(),
                        threshold=np.array(threshold.cpu()),
                    )

            elif self.ood_detection_type.startswith("softmax"):
                softmax_confidence = local_logits.max(dim=1)[0]

                if self.ood_detection_type == "softmax_mean":
                    threshold = softmax_confidence.mean()
                elif self.ood_detection_type == "softmax_median":
                    threshold = softmax_confidence.median()
                else:
                    raise NotImplementedError

                id_indices = (
                    (softmax_confidence >= threshold).nonzero().squeeze(1).cpu()
                )

                if self.current_client_id == 0:
                    os.makedirs(f"tmp/{self.datetime}", exist_ok=True)
                    np.savez(
                        f"tmp/{self.datetime}/{self.round}.npz",
                        softmax_confidence=softmax_confidence.cpu().numpy(),
                        target=torch.stack(local_targets).cpu().numpy(),
                        threshold=np.array(threshold.cpu()),
                    )

            local_logits = local_logits[id_indices]
            local_indices = local_indices[id_indices]

        else:  # for logging (same as DSFL)
            if self.current_client_id == 0:
                negative_energy = torch.log(
                    torch.exp(torch.stack(local_outputs)).sum(dim=1)
                )
                softmax_confidence = local_logits.max(dim=1)[0]
                os.makedirs(f"tmp/{self.datetime}", exist_ok=True)
                np.savez(
                    f"tmp/{self.datetime}/{self.round}.npz",
                    negative_energy=negative_energy.cpu().numpy(),
                    softmax_confidence=softmax_confidence.cpu().numpy(),
                    target=torch.stack(local_targets).cpu().numpy(),
                )

        return [local_logits, local_indices]


class DSFLPlusServerHandler(DSFLServerHandler):
    "DSFL+ server handler."
