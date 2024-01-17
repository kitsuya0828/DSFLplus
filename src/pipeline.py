from logging import Logger

import torch
from fedlab.core.standalone import StandalonePipeline
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


class BasePipeline(StandalonePipeline):
    def __init__(
        self,
        handler,
        trainer,
        test_loader: Dataset,
        logger: Logger,
        writer: SummaryWriter,
    ):
        super().__init__(handler, trainer)
        self.test_loader = test_loader
        self.logger = logger
        self.writer = writer

    def main(self):
        raise NotImplementedError("Please implement main function.")


class DSFLPipeline(BasePipeline):
    def main(self):
        t = 0
        cost = 0
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package

            for b in broadcast:
                if torch.is_tensor(b):
                    cost += (
                        (b.element_size() * b.nelement())
                        * len(sampled_clients)
                        / (1024**3)
                    )  # GB

            # client side
            self.trainer.local_process(
                payload=broadcast, id_list=sampled_clients, round=t
            )
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)
                for p in pack:
                    if torch.is_tensor(p):
                        cost += (p.element_size() * p.nelement()) / (1024**3)  # GB

            self.logger.info(f"Round {t:>3}, Cost {cost:.4f} GB")
            self.writer.add_scalar("Cost", cost, t)

            # evaluate server
            server_loss, server_acc = self.handler.evaluate(self.test_loader)
            self.logger.info(
                f"[Server] Round {t:>3}, Loss {server_loss:.4f}, Test Accuracy {server_acc:.4f}"
            )
            self.writer.add_scalar("Loss/Server", server_loss, t)
            self.writer.add_scalar("Accuracy/Server", server_acc, t)

            t += 1


class DSFLPlusPipeline(DSFLPipeline):
    def main(self):
        super().main()