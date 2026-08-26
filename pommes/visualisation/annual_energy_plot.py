"""Module for annual energy plots."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pommes import generate_summary_dataframes_from_results


def plot_annual_energy(
    solution_path: str,
    resource: str,
    plot_choice: str,
    areas: list[str] | None = None,
    years: list[int] | None = None,
    title: str = "Annual energy production",
    save: bool = True,
    save_name: str = "annual_energy",
    rename: bool = False,
    rename_dict: dict | None = None,
    color_dict: dict | None = None,
) -> None:
    """
    Plot the produced energy by technology for the selected resource.

    Args:
        solution_path (str):
            Path to the solution files containing model results.
        resource (str):
            The energy resource for which production is plotted.
        plot_choice (str):
            Selection mode for the plot, either 'area' or 'year'.
            - 'area': Compares production across different areas in a given
                year.
            - 'year': Compares production across different years in a given
                area.
        areas (list[str] | None, optional):
            List of areas to include in the plot. If None, all areas are
            considered.
        years (list[int] | None, optional):
            List of years to include in the plot. If None, all years are
            considered.
        title (str, optional):
            Title of the plot. Defaults to "Annual energy production".
        save (bool, optional):
            If True, saves the figure as an image. Defaults to True.
        save_name (str, optional):
            Filename for saving the figure. Defaults to "Annual energy".
        rename (bool, optional):
            If True, applies renaming of technologies using rename_dict.
            Defaults to False.
        rename_dict (dict | None, optional):
            Dictionary mapping original technology names to new names.
            Defaults to None.
        color_dict (dict | None, optional):
            Dictionary specifying colors for technologies. Defaults to None.

    Returns:
        None: Displays the plot and optionally saves it as an image.

    Raises:
        ValueError: If `plot_choice` is not 'area' or 'year'.
        ValueError: If `plot_choice` is 'area' but multiple years are selected.
        ValueError: If `plot_choice` is 'year' but multiple areas are selected.
    """
    if plot_choice not in ["area", "year"]:
        raise ValueError("you have to select either 'area' or 'year'.")

    solution = xr.open_dataset(solution_path + "solution.nc")
    parameters = xr.open_dataset(solution_path + "input.nc")
    summary = generate_summary_dataframes_from_results(solution, parameters)
    df_prod = summary["Production - MWh"]

    if years is None:
        years = (
            df_prod.loc[
                (slice(None), slice(None), slice(None), resource, slice(None))
            ]
            .index.get_level_values("year_op")
            .unique()
            .to_list()
        )
    if areas is None:
        areas = (
            df_prod.loc[
                (slice(None), slice(None), slice(None), resource, slice(None))
            ]
            .index.get_level_values("area")
            .unique()
            .to_list()
        )
    if plot_choice == "area":
        not_choice = "year"
        if len(years) > 1:
            raise ValueError(
                "if you select 'area' for plot choice, "
                "you can only choose one year."
            )
    else:
        not_choice = "area"
        if len(areas) > 1:
            raise ValueError(
                "if you select 'year' for plot choice, "
                "you can only choose one area."
            )
    index_dic = {"area": "area", "year": "year_op"}
    label_dic = {"area": areas, "year": years}

    resource_tech_list = []
    for tech in (
        df_prod.loc[
            ("conversion", slice(None), slice(None), resource, slice(None))
        ]
        .index.get_level_values("tech")
        .unique()
        .tolist()
    ):
        if (
            parameters.conversion_factor.sel(
                {"conversion_tech": tech, "resource": resource}
            ).values
            > 0
        ):
            resource_tech_list.append(tech)

    df_selected = df_prod.loc[
        ("conversion", areas, resource_tech_list, resource, years)
    ].reset_index(["type", "resource", index_dic[not_choice]], drop=True)
    df_selected.loc[df_selected["net_generation"] < 1e-6] = 0

    df_selected = (
        df_selected.reset_index()
        .pivot(
            columns="tech",
            values="net_generation",
            index=index_dic[plot_choice],
        )
        .rename(columns=rename_dict)
        .fillna(0)
    )
    tech_list_renamed = df_selected.columns.to_list()

    if color_dict is None:
        col = plt.cm.tab10
        color_dict_renamed = {}
        for k, tech in enumerate(tech_list_renamed):
            color_dict_renamed[tech] = col(k % 10)
    else:
        if rename is True:
            color_dict_renamed = {}
            for key in color_dict:
                color_dict_renamed[rename_dict[key]] = color_dict[key]
        else:
            color_dict_renamed = color_dict

    fig, ax = plt.subplots()
    width = 0.40
    labels = label_dic[plot_choice]
    x = np.arange(len(labels))

    energy_generation = []
    l_bottom = np.zeros(len(labels))
    for n, tech in enumerate(tech_list_renamed):
        energy_generation.append(
            [val / 1e-6 for val in df_selected[tech].to_list()]
        )
        ax.bar(
            x,
            energy_generation[n],
            width,
            color=color_dict_renamed[tech],
            bottom=l_bottom,
            label=tech_list_renamed[n],
            zorder=2,
        )
        l_bottom = [i + j for i, j in zip(l_bottom, energy_generation[n])]

    ax.grid(axis="y", alpha=0.5, zorder=1)
    ax.set_ylim([0, max(l_bottom) * 1.1])
    ax.set_ylabel("Produced energy (TWh)")
    ax.set_title(title)
    plt.xticks(x, labels, rotation=30)
    box = ax.get_position()
    ax.set_position(
        [box.x0, box.y0 + 0.05, box.width * 0.71, box.height * 0.95]
    )
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(np.arange(len(tech_list_renamed)), reverse=True)
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    if save:
        plt.savefig(solution_path + save_name + ".png")
    plt.show()

    return
