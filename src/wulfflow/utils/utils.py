# Utils.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.atoms import Atoms
from ase.geometry import get_layers
from ase.io import read, write

if TYPE_CHECKING:
    import pandas as pd


def numpy_encoder(obj):
    if isinstance(obj, np.integer | np.int64):
        return int(obj)
    elif isinstance(obj, np.floating | np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def quacc_to_json(
    labels: list[str], results: list[dict[str, Any]], filename: str = "results.json"
) -> None:
    def get_atoms_info(atoms: Atoms) -> dict[str, Any]:
        if atoms is None:
            return None
        atom_counts = find_element_numbers(atoms)
        info = {
            "symbols": atoms.get_chemical_symbols(),
            "pbc": atoms.get_pbc().tolist(),
            "cell": atoms.get_cell().tolist(),
            "ChemicalFormula": atoms.get_chemical_formula(),
            "AtomCounts": {k: int(v) for k, v in atom_counts.items()},
        }
        if "miller_index" in atoms.info:
            info["miller_index"] = atoms.info["miller_index"]
        return info

    def get_positions(result: dict[str, Any]) -> list[list[float]]:
        if "molecule" in result:
            return [site.coords.tolist() for site in result["molecule"].sites]
        elif "structure" in result:
            return [site.coords.tolist() for site in result["structure"].sites]
        elif "atoms" in result:
            return result["atoms"].positions.tolist()
        return None

    serialised_results = [
        {
            "label": label,
            "atoms": get_atoms_info(result.get("atoms")),
            "results": result.get("results"),
            "positions": get_positions(result),
            "input_atoms": get_atoms_info(result.get("input_atoms", {}).get("atoms")),
            "density": result.get("density"),
            "elements": [str(el) for el in result.get("elements", [])],
            "formula_pretty": result.get("formula_pretty"),
        }
        for label, result in zip(labels, results, strict=False)
    ]

    with Path(filename).open("w") as f:
        json.dump(serialised_results, f, indent=2, default=numpy_encoder)


def json_to_atoms(filename: str, target_label: str) -> Atoms:
    with Path(filename).open("r") as f:
        data = json.load(f)

    entry = next((e for e in data if e["label"] == target_label), None)
    if entry is None:
        raise ValueError(f"No entry found for label: {target_label}")

    atoms_info = entry["atoms"]
    atoms = Atoms(
        symbols=atoms_info["symbols"],
        positions=entry["positions"],
        cell=atoms_info["cell"],
        pbc=atoms_info["pbc"],
    )
    atoms.info.update(entry["results"])
    atoms.info["ChemicalFormula"] = atoms_info.get(
        "ChemicalFormula", atoms.get_chemical_formula()
    )
    atoms.info["AtomCounts"] = atoms_info.get("AtomCounts", find_element_numbers(atoms))

    miller_index = atoms_info.get("miller_index")
    if miller_index is not None:
        atoms.info["miller_index"] = tuple(miller_index)

    return atoms


def save_structure(ase_atoms: Atoms, formula: str, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    xyz_filename = output_path / f"{formula}.xyz"
    write(str(xyz_filename), ase_atoms, format="extxyz")
    print(f"Saved {xyz_filename}")


def save_results(results: dict[str, tuple[dict[str, Atoms], Any]]) -> None:
    for name, (structures, results_future) in results.items():
        results_list = results_future.result()
        formula_names = list(structures.keys())
        quacc_to_json(formula_names, results_list, f"{name}.json")


def find_element_numbers(atoms: Atoms) -> dict[str, int]:
    chem_elements_list = atoms.get_chemical_symbols()
    element, counts = np.unique(chem_elements_list, return_counts=True)
    return dict(zip(element, counts, strict=False))


def save_surface_energies_to_json(
    surface_energies: pd.DataFrame, filename: str = "surface_energies.json"
) -> None:
    json_data = surface_energies.to_dict(orient="records")

    output_file = Path(filename)
    with output_file.open("w") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False, default=numpy_encoder)

    print(f"Surface energies saved to {output_file}")


def read_existing_slabs(
    slab_dir: str = "slabs", spin: bool = True
) -> dict[str, Atoms] | None:
    """
    Read existing slab structures from a directory.

    Parameters
    ----------
    slab_dir : str
        Directory containing slab structure files
    spin : bool
        Whether to set initial magnetic moments on slabs

    Returns
    -------
    Optional[Dict[str, Atoms]]
        Dictionary of slab structures if directory exists and contains files,
        None otherwise
    """
    slab_path = Path(slab_dir)
    if not slab_path.exists() or not slab_path.is_dir():
        print(f"Slab directory '{slab_dir}' not found")
        return None

    slab_files = list(slab_path.glob("*.xyz"))
    if not slab_files:
        print(f"No .xyz files found in '{slab_dir}'")
        return None

    slabs = {}
    for slab_file in slab_files:
        try:
            atoms = read(slab_file)
            if spin:
                atoms.set_initial_magnetic_moments([0.1] * len(atoms))
            slabs[slab_file.stem] = atoms
        except Exception as e:
            print(f"Error reading {slab_file}: {e}")
            continue

    if not slabs:
        print("No valid slab structures could be read")
        return None

    print(f"Successfully read {len(slabs)} slab structures from {slab_dir}")
    return slabs


def get_energy_miller_atoms(analysis: dict, directory: str) -> tuple[list, list, list]:
    """
    Get miller indices, surface energies, and slab atoms from analysis data for the slabs used
    in constructing the wulff shape.

    Args:
        analysis: dictionary that includes the information about the slabs
        directory: Path to the json files

    Returns:
        Tuple of lists of miller indices, surface energies and slab atoms.

    """
    with open(os.path.join(directory, "slabs.json")) as f:
        slabs_data = json.load(f)

    miller_list = []
    energy_list = []
    slab_atoms_list = []

    for surface in analysis["most_stable_by_miller"]:
        slab_name = surface["MostStableSlabName"]
        miller_tuple = tuple(surface["MillerIndex"])
        energy = float(surface["LowestSurfaceEnergy(J/m²)"])

        for slab in slabs_data:
            if slab["label"] == slab_name:
                slab_atoms = json_to_atoms(
                    os.path.join(directory, "slabs.json"), slab["label"]
                )
                slab_atoms_list.append(slab_atoms)
                break

        miller_list.append(miller_tuple)
        energy_list.append(energy)
    return miller_list, energy_list, slab_atoms_list


def analyse_depth(
    directory: str, analysis: dict, bins=10, range_values=None
) -> dict[str, dict[int, list[tuple[str, float]]]]:
    """
    Analyze the composition along z of the crystal surface.

    Args:
        directory: Path to the json files
        analysis: dictionary that includes the information about the slabs

    Returns:
        Dictionary containing layer-wise composition for each Miller index
    """
    if range_values is None:
        range_values = [0, 50]
    miller_list, _, slab_atoms = get_energy_miller_atoms(analysis, directory)
    data = {}
    for i in range(len(slab_atoms)):
        histogram = {}
        unique = np.unique(slab_atoms[i].get_chemical_symbols())
        for element in unique:
            histo = np.histogram(
                slab_atoms[i].get_positions()[
                    np.where(np.array(slab_atoms[i].get_chemical_symbols()) == element)
                ][:, 2],
                bins=bins,
                range=range_values,
            )
            histogram[element] = [x.tolist() for x in histo]
        miller_key = "".join(str(miller_list[i]))
        data[miller_key] = histogram

    return data


def calculate_average_depth(data):
    """
    Calculate the weighted average depth values across all surfaces for each element.

    Args:
        data (dict): The input JSON data containing depth information

    Returns:
        dict: A dictionary containing average depth values for each element
    """
    depth_data = data["depth"]
    weights = data["area_fraction"]

    element_values = {}
    element_weights = {}

    for surface, elements in depth_data.items():
        for element, (values, bins) in elements.items():
            if element not in element_values:
                element_values[element] = []
                element_weights[element] = []
            element_values[element].append(values)
            element_weights[element].append(weights[surface])

    averages = {}
    for element, value_lists in element_values.items():
        value_array = np.array(value_lists)
        weight_array = np.array(element_weights[element])
        normalized_weights = weight_array / np.sum(weight_array)
        average = np.sum(
            value_array * normalized_weights[:, np.newaxis], axis=0
        ).tolist()
        averages[element] = average

    first_surface = next(iter(depth_data.keys()))
    first_element = next(iter(depth_data[first_surface].keys()))
    bins = depth_data[first_surface][first_element][1]
    return {element: [values, bins] for element, values in averages.items()}


def analyse_crystal_layers(
    directory: str, analysis: dict
) -> dict[str, dict[int, list[tuple[str, float]]]]:
    """
    Analyze the composition of crystal layers for given Miller indices.

    Args:
        directory: Path to the json files
        analysis: dictionary that includes the information about the slabs

    Returns:
        Dictionary containing layer-wise composition for each Miller index
    """
    miller_list, _, slab_atoms = get_energy_miller_atoms(analysis, directory)
    data = {}

    for i in range(len(slab_atoms)):
        layers, _ = get_layers(slab_atoms[i], miller_list[i])
        layer_compositions = analyse_layer_composition(slab_atoms[i], layers)

        miller_key = "".join(str(miller_list[i]))
        data[miller_key] = layer_compositions

    return data


def analyse_layer_composition(
    structure_file: Atoms, layers: np.ndarray, num_layers: int = 1
) -> dict[int, list[tuple[str, float]]]:
    """
    Analyse chemical composition of each layer in the crystal.

    Args:
        structure_file: ASE Atoms object containing crystal structure
        layers: Array containing layer assignments
        num_layers: Number of layers to analyse

    Returns:
        Dictionary containing composition ratios for each layer,
        sorted by ratio in descending order
    """
    layer_compositions = {}

    for layer_idx in range(num_layers):
        layer_atoms = np.where(layers == layer_idx + 1)[0]
        unique_elements, counts = zip(
            *find_element_numbers(structure_file[layer_atoms]).items(), strict=False
        )
        all_elements = structure_file.get_chemical_symbols()
        for element in all_elements:
            if element not in unique_elements:
                unique_elements = np.append(unique_elements, element)
                counts = np.append(counts, 0)
        total_atoms = np.sum(counts)

        if total_atoms == 0:
            composition = [(element, 0) for element in unique_elements]
        else:
            composition = [
                (element, count / total_atoms)
                for element, count in zip(unique_elements, counts, strict=False)
            ]

        composition.sort(key=lambda x: x[1], reverse=True)
        composition.append(("Total", int(total_atoms)))
        layer_compositions[layer_idx] = composition
    return layer_compositions


def save_wulff_data_to_json(wulffdata: dict, filename: str = "wulff_data.json") -> None:
    output_file = Path(filename)
    with output_file.open("w") as f:
        json.dump(wulffdata, f, indent=4)
    print(f"Wulff_data saved to {output_file}")
