"""Module to read solution from previous optimisation."""

import os

import pandas as pd
import xarray as xr
from pandas import DataFrame
from xarray import Dataset


def read_csv_files(
    directory: str, index: bool = False
) -> dict[str, DataFrame]:
    """
    Read CSV files from a directory and return a dictionary of DataFrames.

    Args:
        directory (str): The path to the directory containing CSV files.
        index (bool): Whether to set the first column as the index.
            Default is False.

    Returns:
        Dict[str, pandas.DataFrame]: A dictionary where keys are DataFrame
            names and values are DataFrames read from CSV files.
    """
    files = os.listdir(directory)
    csv_files = [file for file in files if file.endswith(".csv")]

    dataframes = {}

    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        dataframe_name = os.path.splitext(csv_file)[0]

        df = pd.read_csv(file_path, sep=";")

        if len(df.columns) > 2:
            if index:
                df.drop(columns=df.columns[0], inplace=True)

            df.set_index(df.columns[:-1].tolist(), inplace=True)

        dataframes[dataframe_name] = df

    return dataframes


def read_solution(
    directory_variables_path: str, index: bool = False
) -> Dataset:
    """
    Read CSV files from a variables directory and return a merged Dataset.

    Args:
        directory_variables_path (str): The path to the directory containing
            CSV files for variables.
        index (bool): Whether to set the first column as the index.
            Default is False.

    Returns:
        xarray.Dataset: Merged xarray.Dataset for variables.
    """
    dfs_var = read_csv_files(directory_variables_path, index=index)
    s = xr.merge([df.to_xarray() for df in dfs_var.values()])
    return s


def read_dual(directory_constraints_path: str, index: bool = False) -> Dataset:
    """
    Read CSV files from a constraints directory and return merged Dataset.

    Args:
        directory_constraints_path (str): The path to the directory containing
            CSV files for constraints.
        index (bool): Whether to set the first column as the index.
            Default is False.

    Returns:
        xarray.Dataset: Merged xarray.Dataset for constraints.
    """
    dfs_cons = read_csv_files(directory_constraints_path, index=index)
    d = xr.merge([df.to_xarray() for df in dfs_cons.values()])
    return d
