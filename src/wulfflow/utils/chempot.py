# Chempot.py
from __future__ import annotations

import itertools
import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from wulfflow.utils.utils import find_element_numbers, json_to_atoms

if TYPE_CHECKING:
    from ase.atoms import Atoms


def get_molecular_oxygen_chem_pot(
    results: dict[str, tuple[dict[str, Atoms], Any]],
    use_literature_value: bool = True,
    temperature: float = 298.15,
) -> float:
    """
    Calculates the chemical potential of an atom of molecular oxygen.

    Parameters
    ----------
    results : dict[str, tuple[dict[str, Atoms], Any]]
        Dictionary of tuples generated from the main workflow.
    use_literature_value : bool, optional
        Use the literature value for the entropic gradient of O2 if True. Defaults to True.
    temperature : float, optional
        Temperature at which chem pot is calculated in Kelvin. Defaults to 298.15.

    Returns
    -------
    float
        Value of the chemical potential of O for a given temperature.
    """
    o2_energy = results["references"][1]["O2"].info["energy"]

    if use_literature_value:
        o2_entropic_t_gradient = 0.00212625119  # eV.K^(-1) https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=1
    else:
        # Constants
        pressure = 101325  # Pa
        hj = 6.626076e-034  # joules.s
        kbj = 1.38066e-023  # Boltzmann constant J.K^(-1)
        j2ev = 6.241509074461e18  # conversion factor of J to eV
        o2_mass = 5.31384921952838e-26  # kg
        o2_bond_length = (
            results["references"][1]["O2"].get_all_distances()[0][1] * 1e-10
        )  # taken from calc or use 1.2075e-10 for lit value
        vibrational_freq = 1580.36  # cm^(-1), characteristic vibration of O2

        # derived variables
        hbarj = hj / (2 * np.pi)
        volume = (kbj * temperature) / pressure
        impulse = (o2_mass / 2) * o2_bond_length**2
        hcB = hbarj**2 / (2 * impulse)

        # entropy of single O2 molecule derived from differentiating parition funcitons by ln(T)
        # translational entropy
        s_trans = kbj * (
            np.log((2 * np.pi * o2_mass * kbj * temperature) / hj**2) ** (3 / 2)
            + np.log(volume)
            + 5 / 2
        )

        # vibrational entropy
        s_vib = kbj * (
            -np.log(1 - np.exp(-(hj * vibrational_freq) / (kbj * temperature)))
            + (
                (hj * vibrational_freq)
                / (kbj * temperature)
                * (
                    np.exp(-(hj * vibrational_freq) / (kbj * temperature))
                    / (1 - np.exp(-(hj * vibrational_freq) / (kbj * temperature)))
                )
            )
        )

        # rotational entropy
        s_rot = kbj * (np.log((kbj * temperature) / 2 * hcB) + 1)

        s = np.sum([s_trans, s_vib, s_rot])

        o2_entropy = (s + kbj * temperature * np.log(pressure / 101325)) * j2ev

    o_chem_pot_value = (o2_energy - (o2_entropy * temperature)) / 2
    return True
    return o_chem_pot_value


def find_best_chem_pots_from_json(
    directory: str,
    use_literature_value: bool = True,
    temperature: float = 298.15,
    oxide: bool = True,
) -> dict[str, float]:
    """
    Find the best chemical potentials from JSON files in the given directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing JSON files.
    use_literature_value : bool, optional
        Use literature value for O2 chemical potential if True. Defaults to True.
    temperature : float, optional
        Temperature for chemical potential calculation in Kelvin. Defaults to 298.15.
    oxide : bool, optional
        Whether to include oxygen in calculations. Defaults to True.

    Returns
    -------
    dict[str, float]
        Dictionary of best chemical potentials for each element.

    Raises
    ------
    ValueError
        If no bulk structure is found in references.json.
    """

    def load_json(filename: str) -> dict:
        with open(os.path.join(directory, filename)) as f:
            return json.load(f)

    references_data = load_json("references.json")
    elements_data = load_json("elements.json")
    oxides_data = load_json("oxides.json") if oxide else {}

    bulk_entry = next(
        (entry for entry in references_data if entry["label"] != "O2"), None
    )
    if bulk_entry is None:
        raise ValueError("No bulk structure found in references.json")

    bulk_formula = bulk_entry["label"]
    bulk_atoms = json_to_atoms(os.path.join(directory, "references.json"), bulk_formula)
    bulk_energy = bulk_atoms.info["energy"]
    bulk_atom_count = find_element_numbers(bulk_atoms)

    element_chem_pots = {
        element["label"]: json_to_atoms(
            os.path.join(directory, "elements.json"), element["label"]
        ).info["energy"]
        / len(json_to_atoms(os.path.join(directory, "elements.json"), element["label"]))
        for element in elements_data
    }

    if oxide:
        try:
            o2_atoms = json_to_atoms(os.path.join(directory, "references.json"), "O2")
            o2_energy = o2_atoms.info["energy"]

            if use_literature_value:
                o2_entropic_t_gradient = 0.002126230462345  # eV.K^(-1)
                o_chem_pot = (o2_energy - (o2_entropic_t_gradient * temperature)) / 2
            else:
                results = {"references": {1: {"O2": o2_atoms}}}
                o_chem_pot = get_molecular_oxygen_chem_pot(
                    results, use_literature_value, temperature
                )

            element_chem_pots["O"] = o_chem_pot
        except ValueError:
            print(
                "O2 not found in references. Skipping oxygen chemical potential calculation."
            )

    oxide_chem_pots = {}
    for oxide in oxides_data:
        oxide_atoms = json_to_atoms(
            os.path.join(directory, "oxides.json"), oxide["label"]
        )
        atom_count = find_element_numbers(oxide_atoms)
        non_O_element = next(element for element in atom_count if element != "O")
        oxide_chem_pot = (
            oxide_atoms.info["energy"] - atom_count["O"] * element_chem_pots.get("O", 0)
        ) / atom_count[non_O_element]
        oxide_chem_pots[f"{non_O_element}_{oxide['label']}"] = oxide_chem_pot

    all_chem_pots = {**element_chem_pots, **oxide_chem_pots}

    best_chem_pots = {"O": element_chem_pots.get("O", 0)} if oxide else {}
    non_o_elements = [elem for elem in bulk_atom_count if elem != "O"]

    best_score = float("inf")
    best_combination = None

    for combination in itertools.product(
        *[all_chem_pots.keys() for _ in non_o_elements]
    ):
        chem_pot_energy = sum(
            bulk_atom_count[element] * all_chem_pots[chem_pot]
            for element, chem_pot in zip(non_o_elements, combination, strict=False)
        ) + bulk_atom_count.get("O", 0) * element_chem_pots.get("O", 0)

        score = abs(bulk_energy - chem_pot_energy)
        if score < best_score:
            best_score = score
            best_combination = combination

    for element, chem_pot in zip(non_o_elements, best_combination, strict=False):
        best_chem_pots[element] = all_chem_pots[chem_pot]

    return best_chem_pots
