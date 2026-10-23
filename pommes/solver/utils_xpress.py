"""Module for xpress configuration."""

import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import xpress as xp
from linopy import Model
from linopy.solvers import (
    Result,
    Solution,
    Status,
    logger,
    maybe_adjust_objective_sign,
    maybe_convert_path,
    safe_get_solution,
    set_int_index,
)

print(f"Xpress version: {xp.getversion()}")


def run_xpress(
    xpress_problem: xp.problem,
    linopy_model: Model,
    io_api: str = "lp",
    log_fn: str | None = None,
    warmstart_fn: str | None = None,
    basis_fn: str | None = None,
    **solver_options: dict[str, Any],
) -> Result:
    """
    Run the Xpress solver on the given optimization problem.

    Args:
        xpress_problem (xpress.problem):
            Xpress optimization problem instance.
        linopy_model (linopy.Model):
            Linopy model corresponding to the problem.
        io_api (str, optional):
            Input-output format for solver interactions. Defaults to "lp".
        log_fn (str | None, optional):
            Path to the log file. Defaults to None.
        warmstart_fn (str | None, optional):
            Path to warmstart file. Defaults to None.
        basis_fn (str | None, optional):
            Path to save/load model basis. Defaults to None.
        **solver_options (dict[str, Any]):
            Additional solver options.

    Returns:
        linopy.solvers.Result:
            The result object containing the solution and solver status.
    """
    CONDITION_MAP = {
        "lp_optimal": "optimal",
        "mip_optimal": "optimal",
        "lp_infeasible": "infeasible",
        "lp_infeas": "infeasible",
        "mip_infeasible": "infeasible",
        "lp_unbounded": "unbounded",
        "mip_unbounded": "unbounded",
    }

    log_fn = maybe_convert_path(log_fn)
    warmstart_fn = maybe_convert_path(warmstart_fn)
    basis_fn = maybe_convert_path(basis_fn)

    xpress_problem.setControl(solver_options)

    if log_fn is not None:
        xpress_problem.setlogfile(log_fn)

    if warmstart_fn:
        xpress_problem.readbasis(warmstart_fn)

    xpress_problem.solve()

    if basis_fn:
        try:
            if os.path.isfile(f"temp/{basis_fn}"):
                warnings.warn(
                    f"File 'temp/{basis_fn}' exists and will be overridden"
                )
            xpress_problem.writebasis(basis_fn)
        except Exception as err:
            logger.info("No model basis stored. Raised error: %s", err)

    condition = xpress_problem.getProbStatusString()
    termination_condition = CONDITION_MAP.get(condition, condition)
    status = Status.from_termination_condition(termination_condition)
    status.legacy_status = condition

    def get_solver_solution() -> Solution:
        """Get solver solution."""
        objective = xpress_problem.getObjVal()

        var = [str(v) for v in xpress_problem.getVariable()]

        sol = pd.Series(
            xpress_problem.getSolution(var), index=var, dtype=float
        )
        sol = set_int_index(sol)

        try:
            dual = [str(d) for d in xpress_problem.getConstraint()]
            dual = pd.Series(
                xpress_problem.getDual(dual), index=dual, dtype=float
            )
            dual = set_int_index(dual)
        except xp.SolverError:
            logger.warning("Dual values of MILP couldn't be parsed")
            dual = pd.Series(dtype=float)

        return Solution(sol, dual, objective)

    solution = safe_get_solution(status, get_solver_solution)
    maybe_adjust_objective_sign(solution, linopy_model.objective.sense, io_api)

    return Result(status, solution, xpress_problem)


def export_xpress_solution_to_linopy(
    result: Result, model: Model
) -> tuple[str, str]:
    """
    Export a Xpress solver solution to a Linopy model.

    Args:
        result (linopy.solvers.Result):
            The result object containing the Xpress solution.
        model (linopy.Model):
            The Linopy model to update with the solver's solution.

    Returns:
        tuple[str, str]:
            A tuple containing the solver status and termination condition.
    """
    result.info()

    model.objective._value = result.solution.objective
    model.status = result.status.status.value
    model.termination_condition = result.status.termination_condition.value
    model.solver_model = result.solver_model

    if not result.status.is_ok:
        return (
            result.status.status.value,
            result.status.termination_condition.value,
        )

    # map solution and dual to original shape which includes missing values
    sol = result.solution.primal.copy()
    sol.loc[-1] = np.nan

    for name, var in model.variables.items():
        idx = np.ravel(var.labels)
        try:
            vals = sol[idx].values.reshape(var.labels.shape)
        except KeyError:
            vals = sol.reindex(idx).values.reshape(var.labels.shape)
        var.solution = xr.DataArray(vals, var.coords)

    if not result.solution.dual.empty:
        dual = result.solution.dual.copy()
        dual.loc[-1] = np.nan

        for name, con in model.constraints.items():
            idx = np.ravel(con.labels)
            try:
                vals = dual[idx].values.reshape(con.labels.shape)
            except KeyError:
                vals = dual.reindex(idx).values.reshape(con.labels.shape)
            con.dual = xr.DataArray(vals, con.labels.coords)

    return (
        result.status.status.value,
        result.status.termination_condition.value,
    )


def update_xpress_problem(
    model: Model, problem: xp.problem, update: dict, copy: bool = True
) -> xp.problem:
    """
    Update an existing Xpress optimization problem with new coefficients.

    Args:
        model (linopy.Model):
            The Linopy model defining the optimization structure.
        problem (xpress.problem):
            The existing Xpress problem instance.
        update (dict):
            Dictionary specifying constraint-variable coefficient updates.
        copy (bool, optional):
            If True, creates a copy of the problem before updating.
            Defaults to True.

    Returns:
        Any:
            The updated Xpress problem instance.
    """
    if copy:
        p = problem.copy()
    else:
        p = problem

    list_constraints = []
    list_variables = []
    list_coefficients = []

    for constraint, constraint_param in update.items():
        constraint_id = int(
            model.constraints[constraint]
            .sel(constraint_param["coords"])
            .labels
        )
        constraint_label = f"c{constraint_id}"
        for variable, variable_param in constraint_param["variables"].items():
            variable_id = int(
                model.variables[variable].sel(variable_param["coords"]).labels
            )
            variable_label = f"x{variable_id}"
            coefficient = variable_param["coefficient"]

            list_constraints.append(constraint_label)
            list_variables.append(variable_label)
            list_coefficients.append(coefficient)

    p.chgmcoef(list_constraints, list_variables, list_coefficients)

    return p
