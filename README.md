[![Build Status](https://travis-ci.com/rlworkgroup/dowel.svg?branch=master)](https://travis-ci.com/rlworkgroup/dowel)
[![codecov](https://codecov.io/gh/rlworkgroup/dowel/branch/master/graph/badge.svg)](https://codecov.io/gh/rlworkgroup/dowel)
[![Docs](https://readthedocs.org/projects/dowel/badge)](http://dowel.readthedocs.org/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/dowel/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/dowel.svg)](https://badge.fury.io/py/dowel)

# dowel

dowel is a little logger for machine learning research.

This fork includes support for logging to Weights and Biases (wandb) via `dowel.WandbOutput`.

## Installation
```shell
pip install dowel
```

## Usage
```python
import dowel
import numpy as np
import wandb
from dowel import logger, tabular


wandb_api_key = 'your_wandb_api_key'
wandb_username = 'your_wandb_username'
dowel.set_wandb_env_keys(wandb_api_key, wandb_username)

wandb.init(
    project='machine-learning',
    config={'batch_size': 32, 'lr': 0.01}
)

logger.add_output(dowel.StdOutput())
logger.add_output(dowel.TensorBoardOutput('tensorboard_logdir'))
logger.add_output(dowel.WandbOutput())

logger.log('Starting up...')
for i in range(1000):
    logger.push_prefix('itr {}'.format(i))
    logger.log('Running training step')

    tabular.record('itr', i)
    tabular.record('loss', 100.0 / (2 + i))

    # Log a wandb table
    data = np.random.rand(10, 2)
    wandb_table = wandb.Table(data=data, columns=['x', 'y'])
    tabular.record('chart', wandb.plot.scatter(wandb_table, 'x', 'y'))

    logger.log(tabular)

    logger.pop_prefix()
    logger.dump_all()

logger.remove_all()
```
