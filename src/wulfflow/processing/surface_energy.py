from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from wulfflow.utils.constants import CONVERSION_FACTOR
from wulfflow.utils.utils import find_element_numbers, json_to_atoms

if TYPE_CHECKING:
    from ase.atoms import Atoms


def get_surface_energy(slab: Atoms, bulk: Atoms, chem_pots: dict[str, float]) -> float:
    """
    Calculate the surface energy of a slab.

    Parameters
    ----------
    slab : Atoms
        The slab structure.
    bulk : Atoms
        The bulk structure.
    chem_pots : dict[str, float]
        Dictionary of chemical potentials for each element.

    Returns
    -------
    float
        Surface energy of the slab.
    """
    surface_area = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
    E_slab = slab.info["energy"]
    E_bulk = bulk.info["energy"]

    slab_element_counts = find_element_numbers(slab)
    bulk_element_counts = find_element_numbers(bulk)

    bulk_units = min(
        slab_element_counts[el] // bulk_element_counts[el] for el in bulk_element_counts
    )

    element_excess = {
        el: slab_element_counts.get(el, 0)
        - (bulk_units * bulk_element_counts.get(el, 0))
        for el in set(slab_element_counts) | set(bulk_element_counts)
    }

    E_chem_pot = sum(element_excess[el] * chem_pots.get(el, 0) for el in element_excess)

    E_surf = (E_slab - (bulk_units * E_bulk) - E_chem_pot) / (2 * surface_area)
    return E_surf * CONVERSION_FACTOR


def calculate_surface_energies_from_json(
    directory: str, chem_pots: dict[str, float], oxide: bool = True
) -> pd.DataFrame:
    """
    Calculate surface energies from JSON files in the given directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing JSON files.
    chem_pots : dict[str, float]
        Dictionary of chemical potentials for each element.
    oxide : bool, optional
        Whether to include oxygen in calculations. Defaults to True.

    Returns
    -------
    pd.DataFrame
        DataFrame containing surface energies and related information.
    """
    with open(os.path.join(directory, "references.json")) as f:
        references_data = json.load(f)
    with open(os.path.join(directory, "slabs.json")) as f:
        slabs_data = json.load(f)

    bulk_entry = next(
        (entry for entry in references_data if entry["label"] != "O2"), None
    )
    if bulk_entry is None:
        raise ValueError("No bulk structure found in references.json")
    bulk_formula = bulk_entry["label"]
    bulk = json_to_atoms(os.path.join(directory, "references.json"), bulk_formula)

    surface_energies = []
    for slab in slabs_data:
        slab_atoms = json_to_atoms(os.path.join(directory, "slabs.json"), slab["label"])
        surface_energy = get_surface_energy(slab_atoms, bulk, chem_pots)
        miller_index = tuple(slab_atoms.info["miller_index"])

        atom_counts = find_element_numbers(slab_atoms)

        surface_energies.append(
            {
                "SlabName": slab["label"],
                "MillerIndex": miller_index,
                "SurfaceEnergy(J/m²)": surface_energy,
                "ChemicalFormula": slab_atoms.get_chemical_formula(),
                "AtomCounts": {k: int(v) for k, v in atom_counts.items()},
                "UsedChemPots": chem_pots,
            }
        )

    return pd.DataFrame(surface_energies)


def analyse_surface_energies(surface_energies: pd.DataFrame) -> dict[str, Any]:
    """
    Analyse the surface energies.

    Parameters
    ----------
    surface_energies : pd.DataFrame
        DataFrame containing surface energies and related information.

    Returns
    -------
    dict[str, Any]
        Dictionary containing analysis results.
    """
    analysis = {
        "total_surfaces": len(surface_energies),
        "average_energy": surface_energies["SurfaceEnergy(J/m²)"].mean(),
        "min_energy": surface_energies["SurfaceEnergy(J/m²)"].min(),
        "max_energy": surface_energies["SurfaceEnergy(J/m²)"].max(),
        "most_stable_surface": surface_energies.loc[
            surface_energies["SurfaceEnergy(J/m²)"].idxmin(), "SlabName"
        ],
        "least_stable_surface": surface_energies.loc[
            surface_energies["SurfaceEnergy(J/m²)"].idxmax(), "SlabName"
        ],
    }

    most_stable_by_miller = surface_energies.groupby("MillerIndex").apply(
        lambda x: x.loc[x["SurfaceEnergy(J/m²)"].idxmin()]
    )[["SlabName", "SurfaceEnergy(J/m²)", "MillerIndex"]]
    most_stable_by_miller = most_stable_by_miller.reset_index(drop=True)
    most_stable_by_miller = most_stable_by_miller.rename(
        columns={
            "SlabName": "MostStableSlabName",
            "SurfaceEnergy(J/m²)": "LowestSurfaceEnergy(J/m²)",
        }
    )

    analysis["most_stable_by_miller"] = most_stable_by_miller.to_dict(orient="records")

    return analysis
