"""Logger module.

This module instantiates a global logger singleton.
"""
from dowel.histogram import Histogram
from dowel.logger import Logger, LoggerWarning, LogOutput
from dowel.simple_outputs import StdOutput, TextOutput
from dowel.tabular_input import TabularInput
from dowel.csv_output import CsvOutput, LazyCsvOutput  # noqa: I100
from dowel.tensor_board_output import TensorBoardOutput
from dowel.wandb.wandb_setup import set_wandb_env_keys
from dowel.wandb.wandb_output import WandbOutput

logger = Logger()
tabular = TabularInput()

__all__ = [
    'Histogram',
    'Logger',
    'CsvOutput',
    'LazyCsvOutput',
    'StdOutput',
    'TextOutput',
    'LogOutput',
    'LoggerWarning',
    'TabularInput',
    'TensorBoardOutput',
    'logger',
    'tabular',
    'set_wandb_env_keys',
    'WandbOutput'
]
