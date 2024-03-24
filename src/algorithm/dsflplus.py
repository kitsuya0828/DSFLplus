from logging import Logger
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from algorithm.dsfl import DSFLSerialClientTrainer, DSFLServerHandler
from dataset import PartitionedDataset


class DSFLPlusSerialClientTrainer(DSFLSerialClientTrainer):
    """DSFL+ client with local SGD solver."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_clients: int,
        state_dict_dir: str,
        logger: Logger,
        ood_detection_score: str,
        ood_detection_threshold_delta: float,
        cuda=False,
        device=None,
        personal=False,
    ) -> None:
        super().__init__(
            model, num_clients, state_dict_dir, logger, cuda, device, personal
        )
        self.ood_detection_score = ood_detection_score
        self.ood_detection_threshold_delta = ood_detection_threshold_delta

    def setup_dataset(self, dataset: PartitionedDataset):
        super().setup_dataset(dataset)
        self.ood_detection_thresholds = []
        for cid in range(self.num_clients):
            class_count = self.dataset.stats_dict[cid]
            self.ood_detection_thresholds.append(
                np.count_nonzero(class_count) / self.dataset.num_classes
            )
        self.prev_score = [np.inf] * self.num_clients

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

        if self.ood_detection_score is not None:
            if self.round > 0:  # update threshold
                self.ood_detection_thresholds[self.current_client_id] = min(
                    self.ood_detection_thresholds[self.current_client_id]
                    + self.ood_detection_threshold_delta,
                    1.0,
                )

            if self.ood_detection_score == "energy":
                energy = -torch.logsumexp(torch.stack(local_outputs), dim=1)

                threshold = energy.quantile(
                    self.ood_detection_thresholds[self.current_client_id]
                )

                # classify input as in-distribution if negative energy is larger than the threshold value
                id_indices = (energy <= threshold).nonzero().squeeze(1).cpu()

            elif self.ood_detection_score == "msp":
                msp = local_logits.max(dim=1)[0]

                threshold = msp.quantile(
                    1 - self.ood_detection_thresholds[self.current_client_id]
                )

                id_indices = (msp >= threshold).nonzero().squeeze(1).cpu()

            elif self.ood_detection_score == "maxlogit":
                maxlogit = torch.stack(local_outputs).max(dim=1)[0]

                threshold = maxlogit.quantile(
                    1 - self.ood_detection_thresholds[self.current_client_id]
                )

                id_indices = (maxlogit >= threshold).nonzero().squeeze(1).cpu()

            elif self.ood_detection_score == "gen":
                probs = local_logits
                M = 10
                gamma = 0.1
                probs_sorted, _ = torch.sort(probs, dim=1)
                probs_sorted_sliced = probs_sorted[:, -M:]
                generalized_entropy = -torch.sum(
                    probs_sorted_sliced**gamma * (1 - probs_sorted_sliced) ** gamma,
                    dim=1,
                )

                threshold = generalized_entropy.quantile(
                    1 - self.ood_detection_thresholds[self.current_client_id]
                )

                id_indices = (
                    (generalized_entropy >= threshold).nonzero().squeeze(1).cpu()
                )

            elif self.ood_detection_score == "random":
                id_indices = torch.randperm(len(local_logits))[
                    : int(
                        len(local_logits)
                        * self.ood_detection_thresholds[self.current_client_id]
                    )
                ].cpu()

            local_logits = local_logits[id_indices]
            local_indices = local_indices[id_indices]

        else:  # for logging (same as DSFL)
            if self.current_client_id == 0:
                energy = -torch.log(torch.exp(torch.stack(local_outputs)).sum(dim=1))
                softmax_confidence = local_logits.max(dim=1)[0]
                np.savez(
                    f"tmp/{self.datetime}/{self.round}.npz",
                    energy=energy.cpu().numpy(),
                    softmax_confidence=softmax_confidence.cpu().numpy(),
                    target=torch.stack(local_targets).cpu().numpy(),
                )

        return [local_logits, local_indices]


class DSFLPlusServerHandler(DSFLServerHandler):
    "DSFL+ server handler."
