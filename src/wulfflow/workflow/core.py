from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import parsl
from quacc import flow, subflow

from wulfflow.processing.wulff_construction import (
    coverage_plot,
    histogram_plots,
    make_wulff_construction,
    plot_wulff_shape,
)
from wulfflow.utils.chempot import find_best_chem_pots_from_json
from wulfflow.utils.config import bulk_relax_job, config, slab_relax_job
from wulfflow.utils.constants import CIF_FILE_PATH, INPUT_DATA_BULK, INPUT_DATA_SLABS
from wulfflow.workflow.generator import (
    generate_slabs,
    generate_stable_elements,
    generate_stable_oxides,
)

if TYPE_CHECKING:
    from ase.atoms import Atoms

logging.disable(logging.CRITICAL)


@flow
def main_workflow(
    oxide: bool = True, slabs: dict[str, Atoms] | None = None, run_bulk: bool = True
) -> dict[str, tuple[dict[str, Atoms], Any]]:
    """
    Main workflow for running calculations.

    Parameters
    ----------
    oxide : bool
        Whether to include oxide calculations.
    slabs : dict[str, Atoms], optional
        Dictionary of slab structures to calculate.
    run_bulk : bool
        Whether to run bulk calculations.

    Returns
    -------
    dict[str, tuple[dict[str, Atoms], Any]]
        Results dictionary containing structures and calculation results.
    """

    @subflow
    def run_slab_calculations(structures: dict[str, Atoms]) -> list[Any]:
        """Run relaxation calculations for slab structures."""
        return [
            slab_relax_job(atoms, input_data=INPUT_DATA_SLABS)
            for atoms in structures.values()
        ]

    @subflow
    def run_bulk_calculations(structures: dict[str, Atoms]) -> list[Any]:
        """Run relaxation calculations for bulk structures."""
        return [
            bulk_relax_job(atoms, input_data=INPUT_DATA_BULK)
            for atoms in structures.values()
        ]

    stable_structures_oxide = generate_stable_oxides() if oxide else {}
    chempot_reference_structures = reference_structures(oxide=oxide)
    stable_structures_element = generate_stable_elements()

    results = {
        "slabs": (slabs, run_slab_calculations(slabs)) if slabs else (None, None)
    }

    if run_bulk:
        results.update(
            {
                "references": (
                    chempot_reference_structures,
                    run_bulk_calculations(chempot_reference_structures),
                ),
                "elements": (
                    stable_structures_element,
                    run_bulk_calculations(stable_structures_element),
                ),
                "oxides": (
                    stable_structures_oxide,
                    run_bulk_calculations(stable_structures_oxide),
                ),
            }
        )
    else:
        results.update(
            {
                "references": (chempot_reference_structures, None),
                "elements": (stable_structures_element, None),
                "oxides": (stable_structures_oxide, None),
            }
        )

    return results


def main(
    spin: bool = False,
    oxide: bool = True,
    slabgen: bool = True,
    run_bulk: bool = True,
    perform_calculations: bool = True,
):
    """
    Main function to run the complete workflow.

    Parameters
    ----------
    spin : bool
        Whether to include spin polarization for slabs.
    perform_calculations : bool
        Whether to perform calculations or just analysis.
    oxide : bool
        Whether to include oxide calculations.
    slabgen : bool
        Whether to generate slab structures.
    run_bulk : bool
        Whether to run bulk calculations.
    """
    directory = os.getcwd()
    slabs = None
    if slabgen:
        slabs = generate_slabs(CIF_FILE_PATH, write_xyzs=True)
    else:
        slabs = read_existing_slabs(spin=spin)

    if perform_calculations:
        parsl.load(config)
        try:
            results = main_workflow(oxide=oxide, slabs=slabs, run_bulk=run_bulk)
            save_results(results)
        finally:
            parsl.dfk().cleanup()
            parsl.clear()

    chem_pots = find_best_chem_pots_from_json(directory, oxide=oxide)
    surface_energies = calculate_surface_energies_from_json(
        directory, chem_pots, oxide=oxide
    )
    analysis = analyse_surface_energies(surface_energies)
    make_wulff_construction(directory, analysis)
    plot_wulff_shape(directory, analysis, plots=[4, 4], figsize=[15, 15])
    coverage_plot(directory, analysis)
    histogram_plots(directory)
    save_surface_energies_to_json(surface_energies)


if __name__ == "__main__":
    main(spin=True, slabgen=False, run_bulk=True)
