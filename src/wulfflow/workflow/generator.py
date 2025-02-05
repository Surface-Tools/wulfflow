from __future__ import annotations

from hashlib import blake2s
from pathlib import Path
from typing import TYPE_CHECKING

from ase.build import molecule
from ase.io import read, write
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.structure import Structure
from pymatgen.core.surface import generate_all_slabs
from pymatgen.io.ase import AseAtomsAdaptor

from wulfflow.utils.constants import API_KEY, CIF_FILE_PATH, SLAB_GENERATOR_FLAGS
from wulfflow.utils.utils import save_structure

if TYPE_CHECKING:
    from ase.atoms import Atoms


def create_unique_label(miller: tuple[int, int, int], slab: Atoms) -> str:
    """
    Create a unique label for a slab.

    Parameters
    ----------
    miller : tuple[int, int, int]
        Miller indices of the slab.
    slab : Atoms
        The slab structure.

    Returns
    -------
    str
        Unique label for the slab.
    """
    z_positions = slab.positions[:, 2]
    termination_atoms = slab[z_positions > max(z_positions) - 0.1]

    miller_str = "".join(map(str, miller))
    termination_formula = termination_atoms.get_chemical_formula()
    position_hash = blake2s(slab.positions.tobytes(), digest_size=8).hexdigest()

    return f"{miller_str}_{termination_formula}_{position_hash}"


def generate_slabs(input_file: str, write_xyzs: bool = False) -> dict[str, Atoms]:
    """
    Generate slabs from an input file.

    Parameters
    ----------
    input_file : str
        Path to the input file.
    write_xyzs : bool, optional
        Whether to write the slabs to XYZ files. Defaults to False.

    Returns
    -------
    dict[str, Atoms]
        Dictionary of generated slabs.
    """
    bulk = read(input_file)
    pmg_structure = AseAtomsAdaptor.get_structure(bulk)
    pmg_slabs = generate_all_slabs(pmg_structure, **SLAB_GENERATOR_FLAGS)

    slab_dictionary = {}

    for pmg_slab in pmg_slabs:
        miller = tuple(pmg_slab.miller_index)
        ase_slab = AseAtomsAdaptor.get_atoms(pmg_slab.get_orthogonal_c_slab())
        slab_name = create_unique_label(miller, ase_slab)

        if slab_name in slab_dictionary:
            print(f"Warning: Duplicate slab name: {slab_name}. Skipping.")
            continue

        ase_slab.info["miller_index"] = miller
        slab_dictionary[slab_name] = ase_slab

    if write_xyzs:
        slab_dir = Path("slabs")
        slab_dir.mkdir(exist_ok=True)
        for key, slab in slab_dictionary.items():
            write(slab_dir / f"{key}.xyz", slab)

    return slab_dictionary


def get_stable_structures(elements: list[str]) -> list[PDEntry]:
    """
    Get stable structures for given elements.

    Parameters
    ----------
    elements : list[str]
        List of element symbols.

    Returns
    -------
    list[PDEntry]
        List of stable phase diagram entries.
    """
    with MPRester(API_KEY) as mpr:
        entries = mpr.get_entries_in_chemsys(elements)
        pd = PhaseDiagram(entries)
        return pd.stable_entries


def generate_stable_elements(output_dir: str = "on_hull_elements") -> dict[str, Atoms]:
    """
    Generate stable elements and save them to the output directory.

    Parameters
    ----------
    output_dir : str, optional
        Directory to save the stable elements. Defaults to "on_hull_elements".

    Returns
    -------
    dict[str, Atoms]
        Dictionary of stable elements.
    """
    structure = Structure.from_file(CIF_FILE_PATH)
    elements = list(
        {
            element.symbol
            for element in structure.composition.elements
            if element.symbol != "O"
        }
    )
    stable_entries = get_stable_structures(elements)

    stable_structures_element = {}

    for entry in stable_entries:
        if len(entry.composition.elements) == 1:
            formula = entry.composition.reduced_formula
            structure = entry.structure
            ase_atoms = AseAtomsAdaptor.get_atoms(structure)
            stable_structures_element[formula] = ase_atoms
            save_structure(ase_atoms, formula, output_dir)

    return stable_structures_element


def generate_stable_oxides(output_dir: str = "on_hull_oxides") -> dict[str, Atoms]:
    """
    Generate stable oxides and save them to the output directory.

    Parameters
    ----------
    output_dir : str, optional
        Directory to save the stable oxides. Defaults to "on_hull_oxides".

    Returns
    -------
    dict[str, Atoms]
        Dictionary of stable oxides.
    """
    structure = Structure.from_file(CIF_FILE_PATH)
    elements = list({element.symbol for element in structure.composition.elements})
    stable_entries = get_stable_structures(elements)

    stable_structures_oxide = {}

    for entry in stable_entries:
        if (
            "O" in entry.composition.reduced_formula
            and len(entry.composition.elements) == 2
        ):
            formula = entry.composition.reduced_formula
            structure = entry.structure
            ase_atoms = AseAtomsAdaptor.get_atoms(structure)
            stable_structures_oxide[formula] = ase_atoms
            save_structure(ase_atoms, formula, output_dir)

    return stable_structures_oxide


def reference_structures(
    output_dir: str = "reference_structures", oxide: bool = True
) -> dict[str, Atoms]:
    """
    Generate reference structures and save them to the output directory.

    Parameters
    ----------
    output_dir : str, optional
        Directory to save the reference structures. Defaults to "reference_structures".
    oxide : bool, optional
        Whether to include oxygen in the reference structures. Defaults to True.

    Returns
    -------
    dict[str, Atoms]
        Dictionary of reference structures.
    """
    structures = {}
    bulk_structure = Structure.from_file(CIF_FILE_PATH)
    bulk_atoms = AseAtomsAdaptor.get_atoms(bulk_structure)
    bulk_formula = bulk_structure.composition.reduced_formula
    structures[bulk_formula] = bulk_atoms
    save_structure(bulk_atoms, bulk_formula, output_dir)

    if oxide:
        o2_molecule = molecule("O2")
        o2_molecule.set_cell([10, 10, 10])
        o2_molecule.center()
        structures["O2"] = o2_molecule
        save_structure(o2_molecule, "O2", output_dir)

    return structures
