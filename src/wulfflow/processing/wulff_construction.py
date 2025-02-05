# Wulff_construction.py
from __future__ import annotations

import json
import os

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Structure
from wulffpack import SingleCrystal

from wulfflow.utils.constants import CIF_FILE_PATH
from wulfflow.utils.utils import (
    analyse_crystal_layers,
    analyse_depth,
    calculate_average_depth,
    get_energy_miller_atoms,
    save_wulff_data_to_json,
)


def make_wulff_construction(directory: str, analysis: dict) -> WulffShape:
    """
    Generate a Wulff shape and json file with the related data.

    Parameters
    ----------
    directory : str
        Path to the json files.
    analysis : dict
        Dictionary that includes the information about the slabs.

    Returns
    -------
    WulffShape
        The generated Wulff shape.
    """

    miller_list, energy_list, slab_atoms = get_energy_miller_atoms(analysis, directory)

    bulk_structure = Structure.from_file(CIF_FILE_PATH)
    wulff_shape = WulffShape(bulk_structure.lattice, miller_list, energy_list)

    attribute_mapping = {
        "anisotropy": "anisotropy",
        "area_fraction": "area_fraction_dict",
        "effective_radius": "effective_radius",
        "miller_area": "miller_area_dict",
        "shape_factor": "shape_factor",
        "surface_area": "surface_area",
        "total_corner_sites": "tot_corner_sites",
        "total_edges": "tot_edges",
        "volume": "volume",
    }

    wulff_data = {}
    for key, attr in attribute_mapping.items():
        try:
            value = getattr(wulff_shape, attr)
            if isinstance(value, dict):
                converted_dict = {}
                for k, v in value.items():
                    new_key = str(k) if isinstance(k, tuple) else k
                    converted_dict[new_key] = v
                wulff_data[key] = converted_dict
            else:
                wulff_data[key] = value
        except AttributeError:
            print(f"Warning: '{attr}' attribute not found in WulffShape instance")
            wulff_data[key] = None

    wulff_data["coverage"] = analyse_crystal_layers(directory, analysis)
    wulff_data["depth"] = analyse_depth(directory, analysis)
    wulff_data["depth"]["average"] = calculate_average_depth(wulff_data)
    wulff_data["surface_energies"] = {
        "".join(str(miller_list[i])): energy_list[i] for i in range(len(energy_list))
    }
    wulff_data["Elements"] = np.unique(
        bulk_structure.to_ase_atoms().get_chemical_symbols()
    ).tolist()
    wulff_data["natoms"] = {
        "".join(str(miller_list[i])): len(slab_atoms[i])
        for i in range(len(miller_list))
    }
    save_wulff_data_to_json(wulff_data)
    return wulff_shape


def create_color_dict(wulff_dict, cmap, normalize_surface=False):
    """
    Create a color dict based on miller indices, surfaces energies or atom presence.

    Parameters
    ----------
    wulff_dict : dict
        Dictionary containing Wulff data.
    cmap : Colormap
        Colormap to use for coloring.
    normalize_surface : bool, optional
        Whether to normalize the surface energies, by default False.

    Returns
    -------
    dict
        Dictionary mapping miller indices to colors.
    """
    if normalize_surface:
        norm = colors.Normalize(vmin=0, vmax=max(wulff_dict.values()))
    else:
        norm = colors.Normalize(
            vmin=min(wulff_dict.values()), vmax=max(wulff_dict.values())
        )

    return {miller: cmap(norm(energy)) for miller, energy in wulff_dict.items()}


def plot_wulff_shape(
    output_directory,
    analysis,
    cif_file_path=CIF_FILE_PATH,
    show_indices=True,
    show_surface_energies=True,
    plots=None,
    figsize=None,
    rotation=None,
):
    """
    Plot the Wulff shape.

    Parameters
    ----------
    output_directory : str
        Directory to save the plot.
    analysis : dict
        Analysis data for plotting.
    cif_file_path : str, optional
        Path to the CIF file, by default CIF_FILE_PATH.
    show_indices : bool, optional
        Whether to show Miller indices, by default True.
    show_surface_energies : bool, optional
        Whether to show surface energies, by default True.
    plots : list, optional
        Plot grid dimensions, by default None.
    figsize : list, optional
        Figure size, by default None.
    rotation : list, optional
        Rotation angles, by default None.
    """

    if rotation is None:
        rotation = [
            (0, 0, 0),
            (0, 45, 0),
            (0, 90, 0),
            (0, 135, 0),
            (45, 0, 0),
            (45, 45, 0),
            (45, 90, 0),
            (45, 135, 0),
            (90, 0, 0),
            (90, 45, 0),
            (90, 90, 0),
            (90, 135, 0),
            (135, 0, 0),
            (135, 45, 0),
            (135, 90, 0),
            (135, 135, 0),
        ]
    if figsize is None:
        figsize = [15, 15]
    if plots is None:
        plots = [4, 4]
    wulff_dict = {
        tuple(surface["MillerIndex"]): float(surface["LowestSurfaceEnergy(J/m²)"])
        for surface in analysis["most_stable_by_miller"]
    }
    bulk_structure = Structure.from_file(cif_file_path).to_ase_atoms()
    particle = SingleCrystal(wulff_dict, primitive_structure=bulk_structure)
    fig, axes = plt.subplots(
        nrows=plots[0],
        ncols=plots[1],
        subplot_kw={"projection": "3d"},
        sharex=True,
        sharey=True,
        figsize=(figsize[0], figsize[1]),
    )

    if plots[0] == 1 and plots[1] == 1:
        axes = np.array([axes])

    cmap = plt.get_cmap("viridis")

    if show_indices or show_surface_energies:
        normalize_surface = not show_indices
        color_dict = create_color_dict(
            wulff_dict, cmap, normalize_surface=normalize_surface
        )

    for i in range(len(axes.flatten())):
        if rotation:
            axes.flatten()[i].view_init(rotation[i][0], rotation[i][1], rotation[i][2])
        particle.make_plot(
            axes.flatten()[i], colors=color_dict, linewidth=0.3, alpha=0.94
        )

    if show_indices:
        legend_elements = [
            plt.Rectangle(
                (0, 0), 1, 1, facecolor=color_dict[miller], edgecolor="none", alpha=0.94
            )
            for miller in wulff_dict
        ]
        legend_labels = [
            f"({miller[0]}{miller[1]}{miller[2]})" for miller in wulff_dict
        ]

        legend = axes.flatten()[-1].legend(
            legend_elements,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1),
            bbox_transform=fig.transFigure,
            ncol=9,
            frameon=True,
            edgecolor="black",
            fontsize=15,
            handlelength=1.8,
            handleheight=0.8,
            borderpad=0.5,
            labelspacing=0.8,
        )
        legend.set_title("Miller Indices", prop={"size": 20, "weight": "bold"})

    elif show_surface_energies:
        vmin, vmax = min(wulff_dict.values()), max(wulff_dict.values())
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes.ravel().tolist()
        )
        cbar.ax.tick_params(labelsize=20)
        ticks = np.linspace(vmin, vmax, 6)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
        cbar.set_label("Surface Energies ($J/m^2$)", fontsize=30)

    output_path = os.path.join(output_directory, "wulff_shape.png")
    plt.savefig(output_path)
    plt.close()


def coverage_plot(
    output_directory,
    analysis,
    cif_file_path=CIF_FILE_PATH,
    element="Li",
    layer="0",
    plots=None,
    figsize=None,
    rotation=None,
):
    """
    Generate subplots of Wulff shapes colored by the element relative presence in a specific layer.

    Parameters
    ----------
    output_directory : str
        Directory to save the plot.
    analysis : dict
        Analysis data for plotting.
    cif_file_path : str, optional
        Path to the CIF file, by default CIF_FILE_PATH.
    element : str, optional
        Element to highlight, by default "Li".
    layer : str, optional
        Layer to analyze, by default "0".
    plots : list, optional
        Plot grid dimensions, by default None.
    figsize : list, optional
        Figure size, by default None.
    rotation : list, optional
        Rotation angles, by default None.
    """

    if rotation is None:
        rotation = [
            (0, 0, 0),
            (0, 45, 0),
            (0, 90, 0),
            (0, 135, 0),
            (45, 0, 0),
            (45, 45, 0),
            (45, 90, 0),
            (45, 135, 0),
            (90, 0, 0),
            (90, 45, 0),
            (90, 90, 0),
            (90, 135, 0),
            (135, 0, 0),
            (135, 45, 0),
            (135, 90, 0),
            (135, 135, 0),
        ]
    if figsize is None:
        figsize = [15, 15]
    if plots is None:
        plots = [4, 4]
    with open("wulff_data.json") as f:
        data = json.load(f)
    wulff_dict = {}
    for surface in data["coverage"]:
        for tup in data["coverage"][surface][layer]:
            if tup[0] == element:
                wulff_dict[surface] = tup[1]

    vmin, vmax = 0, max(wulff_dict.values())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")
    color_dict = {
        eval(miller): mpl.colors.rgb2hex(cmap(norm(value))[:3])
        for miller, value in wulff_dict.items()
    }

    surf_energies = {
        tuple(surface["MillerIndex"]): float(surface["LowestSurfaceEnergy(J/m²)"])
        for surface in analysis["most_stable_by_miller"]
    }
    bulk_structure = Structure.from_file(cif_file_path).to_ase_atoms()
    particle = SingleCrystal(surf_energies, primitive_structure=bulk_structure)
    fig, axes = plt.subplots(
        nrows=plots[0],
        ncols=plots[1],
        subplot_kw={"projection": "3d"},
        sharex=True,
        sharey=True,
        figsize=(figsize[0], figsize[1]),
    )

    if plots[0] == 1 and plots[1] == 1:
        axes = np.array([axes])

    for i in range(len(axes.flatten())):
        if rotation:
            axes.flatten()[i].view_init(rotation[i][0], rotation[i][1], rotation[i][2])
        particle.make_plot(
            axes.flatten()[i], colors=color_dict, linewidth=0.3, alpha=0.94
        )

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes.ravel().tolist()
    )
    cbar.ax.tick_params(labelsize=20)
    ticks = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
    cbar.set_label("Percentage", fontsize=30)
    output_path = os.path.join(output_directory, "percentage_plot.png")
    plt.savefig(output_path)
    plt.close()


def histogram_plots(directory):
    """
    Generate histogram plots for depth analysis.

    Parameters
    ----------
    directory : str
        Directory containing the Wulff data json file.
    """

    with open(os.path.join(directory, "wulff_data.json")) as f:
        data = json.load(f)

    depth = data["depth"]
    unique = data["Elements"]
    per_element = {}
    surfaces = []
    for element in unique:
        tmp = []
        for surface in depth:
            surfaces.append(surface)
            tmp.append(depth[surface][element][0])
            bins = depth[surface][element][1]
            per_element[element] = tmp

    nplot = len(unique)
    fig, axes = plt.subplots(nplot, sharex=True)
    non_zero_surfaces = [
        i for i, (_, value) in enumerate(data["area_fraction"].items()) if value != 0
    ]
    non_zero_surfaces.append(len(surfaces) - 1)
    if nplot == 1:
        axes = np.array([axes])
    for j, ax in enumerate(axes.flat):
        element_data = per_element[unique[j]]
        for i in non_zero_surfaces:
            ax.hist(element_data[i], bins=bins, alpha=0.5, label=surfaces[i])

        ax.set_title(f"Element: {unique[j]}")
        ax.legend()

    plt.xlabel("Angstrom")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(directory, "histogram_per_element.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2
    nplot = len(non_zero_surfaces)
    surfaces = np.array(surfaces)[non_zero_surfaces].tolist()

    fig, axes = plt.subplots(nplot, sharex=True, sharey=True, constrained_layout=True)
    n_rows = int(np.ceil(np.sqrt(nplot)))
    n_cols = int(np.ceil(nplot / n_rows))
    fig.add_gridspec(n_rows, n_cols)
    if nplot == 1:
        axes = np.array([axes])
    for j, ax in enumerate(axes.flat):
        surface_data = depth[surfaces[j]]
        for i in range(len(surface_data)):
            ax.hist(surface_data[unique[i]][0], bins=bins, alpha=0.5, label=unique[i])

        ax.set_title(surfaces[j])

    ax.legend()
    fig.supxlabel("Angstrom")
    fig.supylabel("Count")
    fig.set_size_inches(min(n_cols * 5, 20), min(n_rows * 4, 16))
    output_path = os.path.join(directory, "histogram_per_surface.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
