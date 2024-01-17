import argparse
import logging
import os
import shutil
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from algorithm import (
    DSFLPlusSerialClientTrainer,
    DSFLPlusServerHandler,
    DSFLSerialClientTrainer,
    DSFLServerHandler,
)
from dataset import PartitionedDataset
from model import CNN_MNIST, CNN_FashionMNIST, ResNet18_CIFAR10
from pipeline import DSFLPipeline, DSFLPlusPipeline
from utils import seed_everything


def main(args, logger, date_time, writer):
    seed_everything(args.seed)

    # data
    dataset_root = f"data/{args.task}"
    dataset_path = os.path.join(dataset_root, "partitions", date_time)
    partitioned_dataset = PartitionedDataset(
        root=dataset_root,
        path=dataset_path,
        num_clients=args.total_clients,
        partition=args.partition,
        num_shards_per_client=args.num_shards_per_client,
        dir_alpha=args.dir_alpha,
        task=args.task,
        public_private_split=args.public_private_split,
        public_size=args.public_size,
        private_size=args.private_size,
    )
    # test data
    test_loader = partitioned_dataset.get_dataloader(
        type="test", batch_size=args.test_batch_size
    )
    # data statistics
    client_stats = partitioned_dataset.get_client_stats()
    client_stats.to_csv(f"./logs/{date_time}.csv")

    # model
    match args.task:
        case "cifar10":
            model = ResNet18_CIFAR10()
            server_model = ResNet18_CIFAR10()
        case "mnist":
            model = CNN_MNIST()
            server_model = CNN_MNIST()
        case "fmnist":
            model = CNN_FashionMNIST()
            server_model = CNN_FashionMNIST()
        case _:
            raise ValueError(f"Invalid task name: {args.task}")

    # server handler, client trainer and pipeline
    state_dict_dir = f"/tmp/{date_time}"
    cuda = torch.cuda.is_available()
    if args.algorithm == "dsfl":
        handler = DSFLServerHandler(
            model=server_model,
            global_round=args.com_round,
            sample_ratio=args.sample_ratio,
            cuda=cuda,
            temperature=args.temperature,
            public_size_per_round=args.public_size_per_round,
            logger=logger,
        )
        trainer = DSFLSerialClientTrainer(
            model=model,
            num_clients=args.total_clients,
            cuda=cuda,
            state_dict_dir=state_dict_dir,
            logger=logger,
        )
        handler.setup_kd_optim(args.kd_epochs, args.kd_batch_size, args.kd_lr)
        trainer.setup_optim(args.epochs, args.batch_size, args.lr)
        trainer.setup_kd_optim(args.kd_epochs, args.kd_batch_size, args.kd_lr)
        handler.setup_dataset(dataset=partitioned_dataset)
        trainer.setup_dataset(dataset=partitioned_dataset)

        standalone_pipeline = DSFLPipeline(
            handler=handler,
            trainer=trainer,
            test_loader=test_loader,
            logger=logger,
            writer=writer,
        )
    elif args.algorithm == "dsflplus":
        handler = DSFLPlusServerHandler(
            model=server_model,
            global_round=args.com_round,
            sample_ratio=args.sample_ratio,
            cuda=cuda,
            temperature=args.temperature,
            public_size_per_round=args.public_size_per_round,
            logger=logger,
        )

        trainer = DSFLPlusSerialClientTrainer(
            model=model,
            num_clients=args.total_clients,
            cuda=cuda,
            state_dict_dir=state_dict_dir,
            logger=logger,
            ood_detection_type=args.ood_detection_type,
            ood_detection_schedule=args.ood_detection_schedule,
        )
        handler.setup_kd_optim(args.kd_epochs, args.kd_batch_size, args.kd_lr)
        trainer.setup_optim(args.epochs, args.batch_size, args.lr)
        trainer.setup_kd_optim(args.kd_epochs, args.kd_batch_size, args.kd_lr)
        handler.setup_dataset(dataset=partitioned_dataset)
        trainer.setup_dataset(dataset=partitioned_dataset)
        trainer.setup_datetime(date_time)

        standalone_pipeline = DSFLPlusPipeline(
            handler=handler,
            trainer=trainer,
            test_loader=test_loader,
            logger=logger,
            writer=writer,
        )
    else:
        raise ValueError(f"Invalid algorithm name: {args.algorithm}")

    standalone_pipeline.main()


def clean_up(args, date_time, writer):
    """Clean up temporary files."""
    writer.flush()
    writer.close()
    state_dict_path = f"/tmp/{date_time}"
    if os.path.exists(state_dict_path):
        shutil.rmtree(state_dict_path)
    dataset_path = f"./data/{args.task}/partitions/{date_time}"
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # algorithm
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["dsfl", "dsflplus"],
    )
    # dataset
    parser.add_argument(
        "--task",
        type=str,
        default="cifar10",
        choices=["mnist", "fmnist", "cifar10"],
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="shards",
        choices=["shards", "hetero_dir", "client_inner_dirichlet"],
    )
    parser.add_argument("--num_shards_per_client", type=int, default=2)
    parser.add_argument("--dir_alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--public_private_split",
        type=str,
        default="even_class",
        choices=["even_class", "random_sample"],
    )
    parser.add_argument("--private_size", type=int, default=40000)
    parser.add_argument("--public_size", type=int, default=10000)
    parser.add_argument("--public_size_per_round", type=int, default=1000)
    # server
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--com_round", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.1)
    # client
    parser.add_argument("--total_clients", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--kd_epochs", type=int, default=5)
    parser.add_argument("--kd_batch_size", type=int, default=100)
    parser.add_argument("--kd_lr", type=float, default=0.1)
    parser.add_argument(
        "--ood_detection_type",
        type=str,
        default=None,
        choices=[
            "energy_mean",
            "energy_median",
            "energy_25percentile",
            "energy_75percentile",
            "energy_schedule",
            "softmax_mean",
            "softmax_median",
        ],
    )
    parser.add_argument(
        "--ood_detection_schedule",
        type=str,
        default=None,
        choices=[
            "linear_25_100",
            "linear_50_100",
            "linear_75_100",
        ],
    )
    # others
    parser.add_argument("--test_batch_size", type=int, default=500)
    parser.add_argument("--comment", type=str, default="")

    args = parser.parse_args()

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    os.makedirs("tmp", exist_ok=True)

    # logging
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger()
    file_handler = logging.FileHandler(f"./logs/{date_time}.log")
    file_handler.setFormatter(
        logging.Formatter("{asctime} [{levelname:.4}] {message}", style="{")
    )
    logger.addHandler(file_handler)

    logger.info(
        "args:\n"
        + "\n".join(
            [f"--{k}={v} \\" for k, v in args.__dict__.items() if v is not None]
        )
    )
    if torch.cuda.is_available():
        logger.info(f"Running on {os.uname()[1]} ({torch.cuda.get_device_name()})")

    writer = SummaryWriter()

    try:
        main(args, logger, date_time, writer)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt")
    except Exception as e:
        logging.exception(e)
    finally:
        clean_up(args, date_time, writer)
