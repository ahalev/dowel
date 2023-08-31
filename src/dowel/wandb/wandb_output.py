import functools
import wandb

from dowel import LogOutput
from dowel import TabularInput



class WandbOutput(LogOutput):
    """
    Weights and biases output for logger.

    Args:
        wandb_run(wandb.run, callable, or None): wandb run. If None, attempts to use global wandb.run.
            If callable, should be a function that takes a log_dir as argument and returns a wandb.run.
        log_dir(str): Save location of wandb_run dir. Ignored if wandb_run is not callable.
    """

    def __init__(self, wandb_run=None, log_dir=None):
        self._wandb_run = wandb_run or wandb.run

        if callable(self._wandb_run):
            self._wandb_run = self._wandb_run(dir=log_dir)

        if self._wandb_run is None:
            raise RuntimeError('wandb.run not found. Call wandb.init before initializing WandbOutput.')

        self._waiting_for_dump = []
        self._default_step = 0

    def record(self, data, prefix=''):
        self._waiting_for_dump.append(
            functools.partial(self._wandb_run.log, data.as_dict)
        )

    def dump(self, step=None):
        for p in self._waiting_for_dump:
            p(step=step or self._default_step)

        self._waiting_for_dump.clear()
        self._default_step += 1

    @property
    def types_accepted(self):
        return TabularInput,
