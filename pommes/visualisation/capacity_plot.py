"""Module for capacity plots."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pommes.io.calculate_summaries import (
    generate_summary_dataframes_from_results,
)


def plot_capacities(
    solution_path: str,
    tech_list: list[str],
    plot_choice: str,
    areas: list[str] | None = None,
    years: list[int] | None = None,
    rename: bool = False,
    rename_dict: dict | None = None,
    title: str = "Production capacities",
    save: bool = True,
    save_name: str = "capacities",
    color_dict: dict | None = None,
) -> None:
    """
    Plot the production capacities of selected technologies.

    Args:
        solution_path (str):
            Path to the solution files containing model results.
        tech_list (list[str]):
            List of technology names to include in the plot.
        plot_choice (str):
            Selection mode for the plot, either 'area' or 'year'.
            - 'area': Compares production capacities across different areas in
                a given year.
            - 'year': Compares production capacities across different years in
                a given area.
        areas (list[str] | None, optional):
            List of areas to include in the plot. If None, all areas are
            considered.
        years (list[int] | None, optional):
            List of years to include in the plot. If None, all years are
            considered.
        rename (bool, optional):
            If True, applies renaming of technologies using rename_dict.
            Defaults to False.
        rename_dict (dict | None, optional):
            Dictionary mapping original technology names to new names.
            Defaults to None.
        title (str, optional):
            Title of the plot. Defaults to "Production capacities".
        save (bool, optional):
            If True, saves the figure as an image. Defaults to True.
        save_name (str, optional):
            Filename for saving the figure. Defaults to "capacities".
        color_dict (dict | None, optional):
            Dictionary specifying colors for technologies. Defaults to None.

    Returns:
        None: Displays the plot and optionally saves it as an image.

    Raises:
        ValueError: If `plot_choice` is not 'area' or 'year'.
        ValueError: If `plot_choice` is 'area' but multiple years are selected.
        ValueError: If `plot_choice` is 'year' but multiple areas are selected.
        ValueError: If no technologies are selected in `tech_list`.
    """
    if plot_choice not in ["area", "year"]:
        raise ValueError("you have to select either 'area' or 'year'.")

    solution = xr.open_dataset(solution_path + "solution.nc")
    parameters = xr.open_dataset(solution_path + "input.nc")
    summary = generate_summary_dataframes_from_results(solution, parameters)
    df_capa = summary["Production capacity - MW"]

    if len(tech_list) == 0:
        raise ValueError(
            "no technologies selected. Select at least one technology."
        )
    if years is None:
        years = (
            df_capa.loc[(slice(None), slice(None), tech_list, slice(None))]
            .index.get_level_values("year_op")
            .unique()
            .to_list()
        )
    if areas is None:
        areas = (
            df_capa.loc[(slice(None), slice(None), tech_list, slice(None))]
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

    type = (
        df_capa.loc[(slice(None), areas[0], tech_list, years[0])]
        .index.get_level_values("type")
        .unique()
        .to_list()
    )
    if len(type) > 1:
        print("Warning, you selected different types of technologies")

    df_selected = df_capa.loc[(type, areas, tech_list, years)].reset_index(
        ["type", index_dic[not_choice]], drop=True
    )
    df_selected = (
        df_selected.reset_index()
        .pivot(
            columns="tech",
            values="power_capacity",
            index=index_dic[plot_choice],
        )
        .rename(columns=rename_dict)
        .fillna(0)
    )
    if rename:
        tech_list_renamed = [rename_dict[tech] for tech in tech_list]
    else:
        tech_list_renamed = tech_list

    if color_dict is None:
        col = plt.cm.tab10
        color_dict_renamed = {}
        for k, tech in enumerate(tech_list_renamed):
            color_dict_renamed[tech] = col(k % 10)
    else:
        if rename:
            color_dict_renamed = {}
            for key in color_dict:
                color_dict_renamed[rename_dict[key]] = color_dict[key]
        else:
            color_dict_renamed = color_dict

    fig, ax = plt.subplots()
    width = 0.40
    labels = label_dic[plot_choice]
    x = np.arange(len(labels))

    capacity = []
    l_bottom = np.zeros(len(labels))
    for n, tech in enumerate(tech_list_renamed):
        capacity.append([val / 1000 for val in df_selected[tech].to_list()])
        ax.bar(
            x,
            capacity[n],
            width,
            bottom=l_bottom,
            label=tech_list_renamed[n],
            color=color_dict_renamed[tech],
            zorder=2,
        )
        l_bottom = [i + j for i, j in zip(l_bottom, capacity[n])]

    ax.grid(axis="y", alpha=0.5, zorder=1)
    ax.set_ylim([0, max(l_bottom) * 1.1])
    ax.set_ylabel("Installed capacity (GW)")
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
