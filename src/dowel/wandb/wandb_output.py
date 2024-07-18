import inspect
import functools

import numpy as np
import wandb
import warnings

from dowel import LogOutput
from dowel import TabularInput
from dowel.utils import colorize


WANDB_TYPES = tuple([
    wandb.viz.CustomChart,
    *(x[1] for x in inspect.getmembers(wandb.data_types, inspect.isclass))
])


class WandbOutput(LogOutput):
    """
    Weights and biases output for logger.

    Args:
        wandb_run(wandb.run, callable, or None): wandb run. If None, attempts to use global wandb.run.
            If callable, should be a function that takes a log_dir as argument and returns a wandb.run.
        log_dir(str): Save location of wandb_run dir. Ignored if wandb_run is not callable.
    """

    def __init__(self, wandb_run=None, log_dir=None, level=0):
        self._wandb_run = wandb_run or wandb.run

        if callable(self._wandb_run):
            self._wandb_run = self._wandb_run(dir=log_dir)

        if self._wandb_run is None:
            raise RuntimeError('wandb.run not found. Call wandb.init before initializing WandbOutput.')

        self.level = level

        self._waiting_for_dump = []
        self._default_step = 0

    def record(self, data, prefix=''):

        wandb_data = dict()

        for k, v in data.as_dict.items():
            if isinstance(v, (np.ScalarType, WANDB_TYPES)):
                wandb_data[k] = v
                data.mark(k)

        self._waiting_for_dump.append(
            functools.partial(self._wandb_run.log, wandb_data)
        )

    def dump(self, step=None):
        if step < 0:
            self._warn(f'Dropping {len(self._waiting_for_dump)} records corresponding to {step=}<0.')
            self._waiting_for_dump.clear()
            return

        for p in self._waiting_for_dump:
            p(step=step or self._default_step)

        self._waiting_for_dump.clear()
        self._default_step += 1

    @property
    def types_accepted(self):
        return TabularInput,

    def _warn(self, msg):
        """Warns the user using warnings.warn.

        The stacklevel parameter needs to be 3 to ensure the call to logger.log
        is the one printed.
        """
        warnings.warn(colorize(msg, 'yellow'),
                      WandbOutputWarning,
                      stacklevel=3)
        return msg


class WandbOutputWarning(UserWarning):
    """Warning class for the TabularInput."""

    pass
