"""A logger that is used to write the logging output."""
import csv
import os
import sys
import wandb

from typing import Any
from typing import Optional
from typing import Dict
from pathlib import Path

class Logger:
    """
    A logger used to write the logs to the standard output and the file.

    Args:
        fpath (optional): The path to the output file to write the log. Defaults to None, in which case, the log will be
            written to the standard output only.
    """
    def __init__(self, fpath: Optional[str] = None) -> None:
        self._console = sys.stdout
        self._file = None
        if fpath is not None:
            self._file = open(fpath, 'w')

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args) -> None:
        self.close()

    def write(self, msg: str) -> None:
        self._console.write(msg)
        if self._file is not None:
            self._file.write(msg)

    def flush(self) -> None:
        self._console.flush()
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self):
        self._console.close()
        if self._file is not None:
            self._file.close()



class BatchLogger:
    """A batch logger.

    Args:
        csv_path: The path to the CSV file to save.
        mode: The type of the mode for the log.
        use_wandb (optional): If True, WanDB will be used.
    """
    def __init__(self, csv_path: str, mode: str='w', use_wandb: bool=False) -> None:
        self._path = csv_path
        self._mode = mode
        self._file = open(csv_path, mode)
        self._is_initialized = False

        # Use Weights and Biases for logging
        self._use_wandb = use_wandb
        if use_wandb:
            self._split = Path(csv_path).stem

    def setup(self, log_dict: Dict[str, Any]) -> None:
        columns = log_dict.keys()
        # Move epoch and batch to the front if in the log_dict
        for key in ['batch', 'epoch']:
            if key in columns:
                columns = [key] + [k for k in columns if k != key]

        self._writer = csv.DictWriter(self._file, fieldnames=columns)
        if self._mode=='w' or (not os.path.exists(self._path)) or os.path.getsize(self._path)==0:
            self._writer.writeheader()
        self._is_initialized = True

    def log(self, log_dict: Dict[str, Any]) -> None:
        if self._is_initialized is False:
            self.setup(log_dict)
        self._writer.writerow(log_dict)
        self.flush()

        if self._use_wandb:
            results = {}
            for key in log_dict:
                new_key = f'{self._split}/{key}'
                results[new_key] = log_dict[key]
            wandb.log(results)

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()