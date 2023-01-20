"""A logger that is used to write the logging output."""
import os
import sys
from typing import Optional

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
