"""Module for energy cost plot."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from pommes import (
    generate_summary_dataframes_from_results,
)


def plot_energy_cost(
    solution_path: str,
    resource: str,
    plot_choice: str,
    areas: list[str] | None = None,
    years: list[int] | None = None,
    title: str = "Energy cost",
    save: bool = True,
    save_name: str = "Energy cost",
) -> None:
    """
    Plot the production cost by cost item for the selected resource.

    Args:
        solution_path (str):
            Path to the solution files containing model results.
        resource (str):
            The energy resource for which production costs are plotted.
        plot_choice (str):
            Selection mode for the plot, either 'area' or 'year'.
            - 'area': Compares costs across different areas in a given year.
            - 'year': Compares costs across different years in a given area.
        areas (list[str] | None, optional):
            List of areas to include in the plot. If None, all areas are
            considered.
        years (list[int] | None, optional):
            List of years to include in the plot. If None, all years are
            considered.
        title (str, optional):
            Title of the plot. Defaults to "Energy cost".
        save (bool, optional):
            If True, saves the figure as an image. Defaults to True.
        save_name (str, optional):
            Filename for saving the figure. Defaults to "Energy cost".

    Returns:
        None: Displays the plot and optionally saves it as an image.

    Raises:
        ValueError: If `plot_choice` is not 'area' or 'year'.
        ValueError: If `plot_choice` is 'area' but multiple years are selected.
        ValueError: If `plot_choice` is 'year' but multiple areas are selected.
    """

    def kg_to_mwh(mass: float) -> float:
        """
        Convert mass in kilograms to energy in megawatt-hours (MWh).

        Args:
            mass (float): Mass in kilograms.

        Returns:
            float: Equivalent energy in MWh.
        """
        return mass * 33.33 / 1000

    def mwh_to_kg(energy: float) -> float:
        """
        Convert energy in megawatt-hours (MWh) to mass in kilograms.

        Args:
            energy (float): Energy in MWh.

        Returns:
            float: Equivalent mass in kilograms.
        """
        return energy / 33.33 / 1000

    if plot_choice not in ["area", "year"]:
        raise ValueError("you have to select either 'area' or 'year'.")

    solution = xr.open_dataset(solution_path + "solution.nc")
    parameters = xr.open_dataset(solution_path + "input.nc")
    dual = xr.open_dataset(solution_path + "dual.nc")
    summary = generate_summary_dataframes_from_results(solution, parameters)
    df_cost = summary["Total costs - EUR"]

    if years is None:
        years = df_cost.index.get_level_values("year_op").unique().to_list()
    if areas is None:
        areas = df_cost.index.get_level_values("area").unique().to_list()
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
    selection = {"area": areas, "year_op": years, "resource": resource}

    conversion_tech_list = []
    for tech in (
        df_cost.loc[(slice(None), "conversion", slice(None), areas, years)]
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
            conversion_tech_list.append(tech)
    conversion = (
        df_cost.loc[
            (slice(None), "conversion", conversion_tech_list, areas, years)
        ]
        .reset_index()
        .groupby(index_dic[plot_choice])
        .sum()
    )

    not_resource = list(solution.operation_net_import_costs.resource.values)
    not_resource.remove(resource)

    conso_other_resource = (
        summary["Production - MWh"]
        .loc[("conversion", areas, conversion_tech_list, not_resource, years)]
        .groupby([plot_choice, "resource"])
        .sum()
    )
    conso_other_resource["resource_price"] = (
        dual.operation_adequacy_constraint.sel(
            {
                key: value
                for key, value in selection.items()
                if key in dual.operation_adequacy_constraint.dims
            }
        )
        .mean("hour")
        .to_dataframe()
        .reset_index(index_dic[not_choice], drop=True)
    )["operation_adequacy_constraint"]
    conso_other_resource["fuel_cost"] = (
        -conso_other_resource["net_generation"]
        * conso_other_resource["resource_price"]
    )
    conso_other_resource = (
        conso_other_resource.reset_index()
        .groupby(index_dic[plot_choice])
        .sum()
    )

    storage_list = parameters.storage_main_resource.to_dataframe()
    storage_tech_list = storage_list.loc[
        storage_list["storage_main_resource"] == resource
    ].index.to_list()
    storage = (
        df_cost.loc[(slice(None), "storage", storage_tech_list, areas, years)]
        .reset_index()
        .groupby(index_dic[plot_choice])
        .sum()
    )

    transport_list = parameters.transport_resource.to_dataframe()
    transport_tech_list = (
        transport_list.loc[transport_list["transport_resource"] == resource]
        .index.get_level_values("transport_tech")
        .unique()
        .to_list()
    )
    transport = (
        df_cost.loc[
            (slice(None), "transport", transport_tech_list, areas, years)
        ]
        .reset_index()
        .groupby(index_dic[plot_choice])
        .sum()
    )

    load_shedding = (
        solution.operation_load_shedding.sel(
            {
                key: value
                for key, value in selection.items()
                if key in solution.operation_load_shedding.dims
            }
        )
        .sum("hour")
        .to_dataframe()
        .reset_index(index_dic[not_choice], drop=True)
    )
    if index_dic[not_choice] in parameters.load_shedding_cost.dims:
        load_shedding["shedding_price"] = (
            parameters.load_shedding_cost.sel(
                {
                    key: value
                    for key, value in selection.items()
                    if key in parameters.load_shedding_cost.dims
                }
            )
            .mean(index_dic[not_choice])
            .values
        )
    else:
        load_shedding["shedding_price"] = parameters.load_shedding_cost.sel(
            {
                key: value
                for key, value in selection.items()
                if key in parameters.load_shedding_cost.dims
            }
        ).values
    load_shedding["shedding_cost"] = (
        load_shedding["operation_load_shedding"]
        * load_shedding["shedding_price"]
    )

    spillage = (
        solution.operation_spillage.sel(
            {
                key: value
                for key, value in selection.items()
                if key in solution.operation_spillage.dims
            }
        )
        .sum("hour")
        .to_dataframe()
        .reset_index(index_dic[not_choice], drop=True)
    )
    if index_dic[not_choice] in parameters.spillage_cost.dims:
        spillage["spillage_price"] = (
            parameters.spillage_cost.sel(
                {
                    key: value
                    for key, value in selection.items()
                    if key in parameters.spillage_cost.dims
                }
            )
            .mean(index_dic[not_choice])
            .values
        )
    else:
        spillage["spillage_price"] = parameters.spillage_cost.sel(
            {
                key: value
                for key, value in selection.items()
                if key in parameters.spillage_cost.dims
            }
        ).values
    spillage["spillage_cost"] = (
        spillage["operation_spillage"] * spillage["spillage_price"]
    )

    df_all_cost = pd.concat(
        [
            conso_other_resource["fuel_cost"],
            spillage["spillage_cost"],
            load_shedding["shedding_cost"],
        ],
        axis=1,
    )
    df_all_cost["conversion_cost"] = conversion["costs"]
    df_all_cost["storage_cost"] = storage["costs"]
    df_all_cost["transport_cost"] = transport["costs"]
    df_all_cost["importation_cost"] = (
        df_cost.loc[(slice(None), "net_import", resource, areas, years)]
        .groupby(index_dic[plot_choice])
        .sum()
    )
    df_all_cost.rename(
        columns={
            "fuel_cost": "Marginal cost",
            "spillage_cost": "Spillage",
            "shedding_cost": "Load shedding",
            "importation_cost": "Imports",
            "conversion_cost": "Conversion \n CAPEX + OPEX",
            "storage_cost": "Storage \n CAPEX + OPEX",
            "transport_cost": "Transport \n CAPEX + OPEX",
        },
        inplace=True,
    )
    item_list = list(df_all_cost.columns)

    total_prod = (
        summary["Production - MWh"]
        .loc[
            ("conversion", slice(None), conversion_tech_list, resource, years)
        ]
        .reset_index()
        .groupby(index_dic[plot_choice])
        .sum()
    )
    total_prod.loc[total_prod["net_generation"] < 100] = 0
    df_all_cost["total_prod"] = total_prod["net_generation"]

    fig, ax = plt.subplots()
    width = 0.40
    labels = label_dic[plot_choice]
    x = np.arange(len(labels))

    energy_cost = []
    l_bottom = np.zeros(len(labels))
    for n, item in enumerate(item_list):
        plot_list = []
        for val, prod in zip(
            df_all_cost[item].to_list(), df_all_cost["total_prod"]
        ):
            if prod == 0:
                plot_list.append(val / prod)
            else:
                plot_list.append(0)
        energy_cost.append(plot_list)
        ax.bar(
            x,
            energy_cost[n],
            width,
            bottom=l_bottom,
            label=list(df_all_cost.columns)[n],
            zorder=2,
        )
        l_bottom = [i + j for i, j in zip(l_bottom, energy_cost[n])]

    ax.grid(axis="y", alpha=0.5, zorder=1)
    ax.set_ylim([0, max(l_bottom) * 1.1])
    ax.set_ylabel("Energy cost (EUR/MWh)")
    ax.set_title(title)
    plt.xticks(x, labels, rotation=30)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(np.arange(len(item_list)), reverse=True)
    box = ax.get_position()
    if resource == "hydrogen":
        second_ax = ax.secondary_yaxis(
            "right", functions=(kg_to_mwh, mwh_to_kg)
        )
        second_ax.set_ylabel("EUR/kg")
        ax.set_position(
            [box.x0, box.y0 + 0.05, box.width * 0.68, box.height * 0.95]
        )
        ax.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc="center left",
            bbox_to_anchor=(1.12, 0.5),
        )
    else:
        ax.set_position(
            [box.x0, box.y0 + 0.05, box.width * 0.73, box.height * 0.95]
        )
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
