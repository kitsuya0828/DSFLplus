format:
	black ./src
	isort .

lint:
	mypy ./src

visualize:
	tensorboard --logdir=src/runs --samples_per_plugin=scalars=5000