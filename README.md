[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

# DS-FL+

The implementation of DS-FL+ (Energy-based Knowledge Distillation for Communication-Efficient Federated Learning, IEICE 2024 student poster session).

## Requirements
* Python version: `3.10.13`
* CUDA version: `11.6`

## Setup

```bash
git clone https://github.com/Kitsuya0828/DSFLplus.git
cd DSFLplus

conda env create -f env.yaml
conda activate dsflplus
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
--sample_ratio=0.1 \
--com_round=10 \
--temperature=0.1 \
--total_clients=100 \
--batch_size=100 \
--epochs=5 \
--lr=0.1 \
--kd_epochs=5 \
--kd_batch_size=100 \
--kd_lr=0.1 \
--ood_detection_type=energy_median \
--test_batch_size=500 \
--comment=example
```
