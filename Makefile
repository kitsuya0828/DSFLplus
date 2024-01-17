format:
	black ./src
	isort .

lint:
	mypy ./src

visualize:
	tensorboard --logdir=src/runs

