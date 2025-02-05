from __future__ import annotations

import hashlib
import json
from functools import partial

import parsl
from parsl.dataflow.memoization import id_for_memo
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SimpleLauncher
from parsl.providers import SlurmProvider
from parsl.utils import get_last_checkpoint
from quacc import job, redecorate
from quacc.recipes.espresso.core import relax_job
from quacc.utils.dicts import remove_dict_keys

from wulfflow.utils.constants import COMMON_PARAMS


def create_command(
    nodes_per_block: int, tasks: int, ndiag: int | None = None, ntg: int | None = None
) -> tuple[str, str]:
    """
    Create the command for running the job.

    Parameters
    ----------
    nodes_per_block : int
        Number of nodes per block.
    tasks : int
        Number of tasks.
    ndiag : int, optional
        Number of diagonal tasks, by default None.
    ntg : int, optional
        Number of task groups, by default None.

    Returns
    -------
    tuple[str, str]
        Pre and post binary commands.
    """
    nodes_per_task = nodes_per_block
    ranks_per_node = (
        COMMON_PARAMS["cores_per_node"] // COMMON_PARAMS["threads_per_process"]
    )
    total_ranks = nodes_per_task * ranks_per_node

    pre_binary = (
        f"srun --mpi=pmix "
        f"--nodes={nodes_per_task} "
        f"--ntasks={total_ranks} "
        f"--ntasks-per-node={ranks_per_node} "
        f"--cpus-per-task={COMMON_PARAMS['threads_per_process']}"
    )

    post_binary = ""
    if ntg:
        post_binary += f" -ntg {ntg}"
    if ndiag:
        post_binary += f" -ndiag {ndiag}"

    return pre_binary, post_binary.strip()


def create_worker_init(command: str) -> str:
    """
    Create the worker initialization script.

    Parameters
    ----------
    command : str
        Command to initialize the worker.

    Returns
    -------
    str
        Worker initialization script.
    """
    return """
source /iridisfs/home/ba3g18/.bashrc
conda activate Quacc_restarts

module purge
module load pmix
module load binutils/2.42
module load intel-mpi/2021.12
module load mkl/2024.1.0
module load intel-compilers/2024.1.0

export OMP_PLACES=cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export I_MPI_FABRICS=shm:ofi
export I_MPI_COLL_INTRANODE=shm
export I_MPI_ADJUST_ALLGATHERV=1
export MKL_THREADING_LAYER=sequential
export ESPRESSO_PSEUDO="/home/ba3g18/Repos/SSSP_1.3.0_PBE_efficiency"
export I_MPI_PMI_LIBRARY=/iridisfs/i6software/slurm/24.05.1/lib/libpmi.so
"""


def create_executor(
    label: str, nodes_per_block: int, max_blocks: int
) -> HighThroughputExecutor:
    """
    Create a HighThroughputExecutor.

    Parameters
    ----------
    label : str
        Label for the executor.
    nodes_per_block : int
        Number of nodes per block.
    max_blocks : int
        Maximum number of blocks.

    Returns
    -------
    HighThroughputExecutor
        The created executor.
    """
    return HighThroughputExecutor(
        label=label,
        max_workers_per_node=1,
        cores_per_worker=1.0e-6,
        provider=SlurmProvider(
            cores_per_node=COMMON_PARAMS["threads_per_process"],
            partition=COMMON_PARAMS["partition"],
            account=COMMON_PARAMS["account"],
            walltime="60:00:00",
            nodes_per_block=nodes_per_block,
            init_blocks=1,
            min_blocks=0,
            max_blocks=max_blocks,
            launcher=SimpleLauncher(),
            worker_init=create_worker_init(create_command(nodes_per_block, 1)[0]),
        ),
    )


def hash_dict(d, exclude_keys: list[str] | None = None) -> bytes:
    """
    Hash a dictionary.

    Parameters
    ----------
    d : dict
        Dictionary to hash.
    exclude_keys : list[str], optional
        Keys to exclude from hashing, by default None.

    Returns
    -------
    bytes
        Hash of the dictionary.
    """
    if exclude_keys is None:
        exclude_keys = []

    for key in exclude_keys:
        d = remove_dict_keys(d, key)

    json_str = json.dumps(d, sort_keys=True)
    return hashlib.md5(json_str.encode("utf-8")).digest()


hash_dict_custom = partial(hash_dict, exclude_keys=["restart_mode"])
id_for_memo.register(dict)(hash_dict_custom)


def create_relax_job(
    label: str,
    nodes_per_block: int,
    max_blocks: int,
    ndiag: int | None = None,
    ntg: int | None = None,
):
    """
    Create a relaxation job.

    Parameters
    ----------
    label : str
        Label for the job.
    nodes_per_block : int
        Number of nodes per block.
    max_blocks : int
        Maximum number of blocks.
    ndiag : int, optional
        Number of diagonal tasks, by default None.
    ntg : int, optional
        Number of task groups, by default None.

    Returns
    -------
    tuple
        Executor and decorated job.
    """
    executor = create_executor(label, nodes_per_block, max_blocks)
    pre_cmd, post_cmd = create_command(nodes_per_block, 1, ndiag=ndiag, ntg=ntg)

    decorated_job = redecorate(
        relax_job,
        job(
            executors=[label],
            cache=True,
            settings_swap={
                "RESTART_MODE": True,
                "ESPRESSO_PARALLEL_CMD": [pre_cmd, post_cmd] if post_cmd else pre_cmd,
            },
        ),
    )
    return executor, decorated_job


slab_executor, slab_relax_job = create_relax_job(
    label="slab_executor", nodes_per_block=2, max_blocks=15
)

bulk_executor, bulk_relax_job = create_relax_job(
    label="bulk_executor", nodes_per_block=2, max_blocks=1
)

config = parsl.Config(
    strategy="htex_auto_scale",
    checkpoint_mode="task_exit",
    checkpoint_period="00:30:00",
    checkpoint_files=get_last_checkpoint(),
    executors=[slab_executor, bulk_executor],
)
