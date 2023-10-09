"""A module that have helper functions for DDP trainings.

References:
[1]. https://theaisummer.com/distributed-training-pytorch/
[2]. https://pytorch.org/docs/stable/elastic/run.html#launcher-api
[3]. https://github.com/The-AI-Summer/pytorch-ddp/blob/main/train_ddp_mixed_presicion.py
[4]. https://pytorch.org/docs/stable/distributed.html#launch-utility (Launch utility section)
"""
import argparse
import builtins as __builtin__
import logging
import os
from datetime import timedelta

import torch
from torch import distributed


def add_ddp_specific_flags(parser: argparse.ArgumentParser) -> None:
    """Adds DDP specfic flags."""
    parser.add_argument("--local_rank", type=int, help="The local rank for DDP training.")
    return parser


def initialize_ddp(local_rank: int) -> None:
    """Initializes the distributed data processing (DDP) training.

    See Ref.[1] on how to launch DDP training.

    Args:
        local_rank: The local rank of each process.
    """
    if not distributed.is_available():
        raise RuntimeError("The distributed pacakage is not available, can't use DDP!")

    if not distributed.is_nccl_available():
        raise RuntimeError("The NCCL backend is not avaible for DDP training!")
    _DEFAULT_DIST_URL = "env://"  # default
    world_size = int(os.environ["WORLD_SIZE"])  # Number of processes particpating in the job.
    rank = int(os.environ["RANK"])  # Rank of the current processes
    distributed.init_process_group(
        backend="nccl", init_method=_DEFAULT_DIST_URL, world_size=world_size, rank=rank, timeout=timedelta(seconds=1800)
    )

    # Set current GPU device - NCCL backend only.
    try:
        torch.cuda.set_device(local_rank)
    except Exception as e:
        logging.error("Can't set device in DDP, consider changing nproc_per_node to #GPUS on your machine!")
        raise e

    # Synchronizes all processes
    distributed.barrier()
    _setup_printing(is_master=local_rank == 0)
    print(f"Initialized DDP with world_size = {world_size}, rank = {rank}, local_rank = {local_rank}.")


def _setup_printing(is_master: bool) -> None:
    """Sets up and allow printing only on the master processes.

    Args:
        is_master: True if this is the master process.
    """
    builtin_print = __builtin__.print

    def wrapped_print(*args, **kwargs) -> None:
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = wrapped_print
