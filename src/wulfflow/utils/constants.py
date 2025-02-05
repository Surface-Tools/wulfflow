# constants.py
from __future__ import annotations

API_KEY = ""
CIF_FILE_PATH = "/scratch/ba3g18/Workflow/Li7La3Zr2O12.cif"

SLAB_GENERATOR_FLAGS = {
    "symmetrize": True,
    "repair": False,
    "primitive": False,
    "max_index": 2,
    "lll_reduce": False,
    "ftol": 1.0e-4,
    "tol": 1.0e-4,
    "center_slab": True,
    "min_slab_size": 15,
    "max_normal_search": 5,
    "in_unit_planes": False,
    "min_vacuum_size": 15,
    "max_broken_bonds": 100000,
}

INPUT_DATA_BULK = {
    "disk_io": "none",
    "upscale": 1e5,
    "conv_thr": 1e-6,
    "mixing_beta": 0.75,
    "startingwfc": "random",
    "etot_conv_thr": 1.0e-4,
    "forc_conv_thr": 1.0e-3,
}

INPUT_DATA_SLABS = {
    "nspin": 1,
    "disk_io": "minimal",
    "upscale": 1e5,
    "conv_thr": 1e-6,
    "bfgs_ndim": 1,
    "mixing_beta": 0.15,
    "restart_mode": "restart",
    "etot_conv_thr": 1.0e-4,
    "forc_conv_thr": 1.0e-3,
    "electron_maxstep": 300,
}

COMMON_PARAMS = {
    "account": "special",
    "partition": "batch",
    "max_walltime": 216000,
    "cores_per_node": 192,
    "threads_per_process": 1,
}

CONVERSION_FACTOR = 16.02
