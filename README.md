[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

# DS-FL+

> [!NOTE]
> The implementation of **DS-FL+** (Energy-based Knowledge Distillation for Communication-Efficient Federated Learning, IEICE 2024 student poster session) is available [here](https://github.com/Kitsuya0828/DSFLplus/tree/v1.0.0).

This is the implementation of **DS-FL+** (Energy-based Thresholding and Knowledge Distillation for Communication-Efficient Federated Learning on Non-IID Data).

## Requirements
* Python version: `3.10.13`
* CUDA version: `11.6`

## Setup

```bash
git clone https://github.com/Kitsuya0828/DSFLplus.git
cd DSFLplus

conda env create -f env.yaml
conda activate dsflplus

# or

# https://pytorch.org/get-started/previous-versions/
conda create -n dsflplus python=3.10.13
conda activate dsflplus
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install black fedlab gdown isort matplotlib mypy pandas plotly tensorboard
```

## Run

```bash
cd src

python main.py --help

# or

python main.py \
--algorithm=dsflplus \
--task=cifar10 \
--partition=shards \
--num_shards_per_client=2 \
--seed=42 \
--public_private_split=even_class \
--private_size=40000 \
--public_size=10000 \
--public_size_per_round=1000 \
--sample_ratio=1.0 \
--com_round=1000 \
--temperature=0.1 \
--total_clients=100 \
--batch_size=100 \
--epochs=5 \
--lr=0.1 \
--kd_epochs=5 \
--kd_batch_size=100 \
--kd_lr=0.1 \
--ood_detection_score=energy \
--ood_detection_threshold_delta=0.0025 \
--test_batch_size=500 \
--comment=shards_energy_delta00025
```
