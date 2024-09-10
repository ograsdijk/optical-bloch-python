from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from numbers import Number
from typing import Sequence

import numpy as np
import numpy.typing as npt
import sympy as smp

try:
    from juliacall import Main as jl

    julia_flag = True
except Exception as e:
    julia_flag = False
    logging.error("juliacall not installed; julia functionality not enabled")

from .generate_code import get_compound_parameter_order


class OdeParameters:
    def __init__(
        self,
        free_parameters: Sequence[smp.Symbol],
        compound_parameters: Sequence[tuple[smp.Symbol, smp.Expr]],
        fixed_parameters: Sequence[tuple[smp.Symbol, Number]],
    ) -> None:
        self._free_parameters = dict([(str(par), par) for par in free_parameters])
        self._compound_parameters = dict(
            [(str(par), par) for par, _ in compound_parameters]
        )
        self._fixed_parameters = dict([(str(par), par) for par, _ in fixed_parameters])

        for key, val in fixed_parameters + compound_parameters:
            setattr(self, str(key), val)
        for key in free_parameters:
            setattr(self, str(key), None)

    def __repr__(self) -> str:
        rep = "OdeParameters("
        for par in self._free_parameters:
            rep += f"{par}={getattr(self, par)}, "
        return rep.strip(", ") + ")"

    def __setattr__(self, name: str, value) -> None:
        if name in [
            "_free_parameters",
            "_compound_parameters",
            "_fixed_parameters",
        ]:
            super(OdeParameters, self).__setattr__(name, value)
        elif name in self._free_parameters.keys():
            assert not isinstance(
                value, smp.Expr
            ), "Cannot change parameter from numeric to str"
            super(OdeParameters, self).__setattr__(name, value)
        elif name in self._compound_parameters.keys():
            assert isinstance(
                value, smp.Expr
            ), "Cannot change parameter from str to numeric"
            super(OdeParameters, self).__setattr__(name, value)
        elif name in self._fixed_parameters.keys():
            assert not isinstance(value, smp.Expr), "Cannot change numeric to str"
            super(OdeParameters, self).__setattr__(name, value)
        else:
            raise AssertionError(
                "Cannot instantiate new parameter on initialized OdeParameters object"
            )

    @property
    def parameter_values(self) -> dict[smp.Symbol, Number | smp.Expr | smp.Symbol]:
        parameter_values: dict[smp.Symbol, Number | smp.Expr | smp.Symbol] = {}
        for par_str, par in self._compound_parameters.items():
            parameter_values[par] = getattr(self, par_str)
        for par_str, par in self._fixed_parameters.items():
            parameter_values[par] = getattr(self, par_str)
        for par_str, par in self._free_parameters.items():
            parameter_values[par] = getattr(self, par_str)
        return parameter_values

    @property
    def fixed_parameters(self) -> dict[smp.Symbol, Number]:
        return dict(
            [(par, getattr(self, str(par))) for par in self._fixed_parameters.values()]
        )

    @property
    def free_parameters(self) -> dict[smp.Symbol, Number]:
        return dict(
            [(par, getattr(self, str(par))) for par in self._free_parameters.values()]
        )

    @property
    def compound_parameters(self) -> dict[smp.Symbol, smp.Expr]:
        return dict(
            [
                (par, getattr(self, str(par)))
                for par in self._compound_parameters.values()
            ]
        )

    @property
    def p(self) -> tuple[Number]:
        return tuple([getattr(self, par) for par in self._free_parameters.keys()])

    def parameter_index(self, parameter: str | smp.Symbol) -> int:
        if isinstance(parameter, str):
            return [str(par) for par in self.free_parameters].index(parameter)
        else:
            return [self.free_parameters].index(parameter)

    def _time_evolution_compound_parameter(
        self, parameter: str, t: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        expression = self.parameter_values[self._compound_parameters[parameter]]

        # loop over to replace parameters, could be more elegant by figuring
        # out the order like when the code is generated, but this works
        while True:
            expression = expression.subs(self.parameter_values)
            repeat = False
            for val in expression.free_symbols:
                if val in self.parameter_values.values():
                    repeat = True
                    break
            if not repeat:
                break

        # vectorize over all function calls dotting (.) function calls, e.g. sin.(t)
        # instead of sin(t)
        pattern = r"([a-zA-Z])\("
        str_expression = re.sub(pattern, r"\1.(", str(expression))

        # vectorize full expression by putting a dot (.) before all mathematical
        # operators
        pattern = r"([+\-*/^=<>])"
        str_expression = re.sub(pattern, r".\1", str_expression)
        jl.t = t
        return np.array(jl.seval(str_expression))

    def time_evolution_parameter(
        self, parameter: str, t: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        if parameter in self._compound_parameters.keys():
            return self._time_evolution_compound_parameter(parameter, t)
        elif parameter in self._fixed_parameters.keys():
            return np.ones(t.shape) * getattr(self, parameter)
        elif parameter in self._free_parameters.keys():
            return np.ones(t.shape) * getattr(self, parameter)
        else:
            raise IndexError(f"parameter {parameter} not found")


@dataclass
class OBEProblem:
    odepars: OdeParameters
    ρ: npt.NDArray[np.complex_]
    tspan: Sequence[float]
    name: str = "prob"


@dataclass
class OBEProblemConfig:
    method: str = "Tsit5()"
    abstol: float = 1e-7
    reltol: float = 1e-4
    dt: float = 1e-8
    callback: str | None = None
    dtmin: int | None = None
    maxiters: int = 100_000
    saveat: list[float] | npt.NDArray[np.float_] | None = None
    save_everystep: bool = True
    save_start: bool = True
    save_idxs: list[int] | None = None
    progress: bool = False


@dataclass
class OBEResult:
    t: npt.NDArray[np.float_]
    y: npt.NDArray[np.complex_]


def get_diagonal_indices_flattened(
    size: int, states: Sequence[int] | None = None, mode: str = "python"
) -> npt.NDArray[np.int_]:
    if states is None:
        ind = list(range(size))
    else:
        ind = list(states)
    indices = np.ravel_multi_index((ind, ind), dims=(size, size))
    if mode == "julia":
        indices += 1
    return indices


def init_julia(lindblad_function: str) -> None:
    jl.seval(lindblad_function)
    jl.seval("using DifferentialEquations")


### Single trajectories


def setup_problem(problem: OBEProblem) -> None:
    jl.p = problem.odepars.p
    jl.ρ = problem.ρ
    # convert to Julia Array, by default it makes a PyCall object
    jl.seval("ρ = convert(Array, ρ)")
    jl.tspan = problem.tspan
    assert jl.seval("@isdefined lindblad!"), "Lindblad function is not defined in Julia"
    jl.seval(f"{problem.name} = ODEProblem(lindblad!,ρ,tspan,p)")


def solve_problem(problem: OBEProblem, config: OBEProblemConfig) -> None:
    method = config.method
    abstol = config.abstol
    reltol = config.reltol
    dt = config.dt
    callback = config.callback
    dtmin = config.dtmin
    maxiters = config.maxiters
    saveat = config.saveat
    progress = config.progress
    save_everystep = config.save_everystep
    save_idxs = config.save_idxs

    force_dtmin = "false" if dtmin is None else "true"
    _dtmin = "0" if dtmin is None else str(dtmin)
    _saveat = "[]" if saveat is None else str(saveat)
    _save_idxs = "nothing" if save_idxs is None else str(save_idxs)

    if callback is not None:
        jl.seval(
            f"""
            sol = solve({problem.name}, {method}, abstol = {abstol},
                        reltol = {reltol}, dt = {dt},
                        progress = {str(progress).lower()},
                        callback = {callback}, saveat = {_saveat},
                        dtmin = {_dtmin}, maxiters = {maxiters},
                        force_dtmin = {force_dtmin}, save_idxs = {_save_idxs},
                        save_everystep = {str(save_everystep).lower()}
                    )
        """
        )
    else:
        jl.seval(
            f"""
            sol = solve({problem.name}, {method}, abstol = {abstol},
                        reltol = {reltol}, dt = {dt},
                        progress = {str(progress).lower()}, saveat = {_saveat},
                        dtmin = {_dtmin}, maxiters = {maxiters},
                        force_dtmin = {force_dtmin}, save_idxs = {_save_idxs},
                        save_everystep = {str(save_everystep).lower()}
                    )
        """
        )


def get_results(solution_name: str = "sol") -> OBEResult:
    t = np.array(jl.sol.t)
    shape = jl.seval(f"size({solution_name})")
    y = np.zeros(shape[::-1], dtype=complex)
    for idx in range(t.size):
        y[idx] = np.array(jl.sol.u[idx])
    return OBEResult(t, y)


def setup_solve_problem(
    problem: OBEProblem, config: OBEProblemConfig = OBEProblemConfig()
) -> OBEResult:
    setup_problem(problem)
    solve_problem(problem, config)
    return get_results()


### Parameter scans


@dataclass
class OBEEnsembleProblem:
    problem: OBEProblem
    parameters: tuple[str]
    scan_values: tuple[Sequence[int | float | complex], ...]
    name: str = "ens_prob"
    output_func: str | None = None
    combinations: bool = True


@dataclass
class OBEEnsembleProblemConfig(OBEProblemConfig):
    distributed_method: str = "EnsembleDistributed()"
    trajectories: int | None = None


def setup_parameter_scan(
    odepars: OdeParameters,
    ensemble_problem: OBEEnsembleProblem,
) -> None:
    values = ensemble_problem.scan_values
    parameters = ensemble_problem.parameters
    combinations = ensemble_problem.combinations

    if not combinations:
        jl.params = np.asarray(values)
        jl.seval("params = convert(Array, params)")
    else:
        jl.params = tuple([tuple(sv) for sv in values])
    jl.seval("@everywhere params = $params")

    if not combinations:
        pars = list(odepars.p)
        for idp, parameter in enumerate(parameters):
            index = odepars.parameter_index(parameter)
            pars[index] = f"params[{idp+1}, i]"

        _pars = "(" + ",".join([str(par) for par in pars]) + ")"

        jl.seval(f"""
        @everywhere function prob_func(prob, i, repeat)
            remake(prob, p = {_pars})
        end
        """)
    else:
        pars = list(odepars.p)
        for idp, parameter in enumerate(parameters):
            index = odepars.parameter_index(parameter)
            pars[index] = f"params[{idp+1}][p{idp+1}]"

        _pars = "(" + ",".join([str(par) for par in pars]) + ")"
        p = ",".join(
            [f"p{idx}" for idx in range(1, len(ensemble_problem.parameters) + 1)]
        )
        dim = tuple([len(v) for v in values])
        jl.seval(f"""
        @everywhere function prob_func(prob, i, repeat)
            {p} = Tuple(CartesianIndices({dim})[i])
            remake(prob, p = {_pars})
        end
        """)


def init_julia_ensemble(lindblad_function: str, nprocs: int) -> None:
    jl.seval("using Distributed")
    if jl.nprocs() < nprocs:
        jl.seval(f"addprocs({nprocs} - nprocs())")
    elif jl.nprocs() > nprocs:
        procs = jl.seval("procs()")
        procs = procs[nprocs:]
        jl.eval(f"rmprocs({procs})")
    jl.seval("@everywhere using DifferentialEquations")
    jl.seval(f"@everywhere {lindblad_function}")


def define_ensemble_problem(ensemble_problem: OBEEnsembleProblem):
    jl.seval("@everywhere ρ = $ρ")
    if ensemble_problem.output_func is None:
        jl.seval(f"""
            {ensemble_problem.name} = EnsembleProblem({ensemble_problem.problem.name}, prob_func = prob_func)
        """)
    else:
        jl.seval(f"""
            {ensemble_problem.name} = EnsembleProblem({ensemble_problem.problem.name}, prob_func = prob_func, output_func = {ensemble_problem.output_func})
        """)


def setup_ensemble_problem(
    odepars: OdeParameters, ensemble_problem: OBEEnsembleProblem
) -> None:
    setup_parameter_scan(odepars, ensemble_problem)
    define_ensemble_problem(ensemble_problem)


def solve_problem_ensemble(
    ensemble_problem: OBEEnsembleProblem, config: OBEEnsembleProblemConfig
) -> None:
    ensemble_problem_name = ensemble_problem.name

    method = config.method
    abstol = config.abstol
    reltol = config.reltol
    dt = config.dt
    callback = config.callback
    saveat = config.saveat
    trajectories = config.trajectories
    save_idxs = config.save_idxs
    distributed_method = config.distributed_method
    save_everystep = config.save_everystep
    save_start = config.save_start

    if not ensemble_problem.combinations:
        _trajectories = "size(params)[2]" if trajectories is None else str(trajectories)
    else:
        _trajectories = (
            f"{np.prod([len(v) for v in ensemble_problem.scan_values])}"
            if trajectories is None
            else str(trajectories)
        )
    _saveat = "[]" if saveat is None else str(saveat)
    _save_idxs = "nothing" if save_idxs is None else str(save_idxs)
    if callback is not None:
        jl.seval(
            f"""
            sol = solve({ensemble_problem_name}, {method}, {distributed_method},
                        abstol = {abstol}, reltol = {reltol}, dt = {dt},
                        trajectories = {_trajectories}, callback = {callback},
                        save_everystep = {str(save_everystep).lower()},
                        saveat = {_saveat}, save_idxs = {_save_idxs},
                        save_start = {str(save_start).lower()}
                    );
        """
        )
    else:
        jl.seval(
            f"""
            sol = solve({ensemble_problem_name}, {method}, {distributed_method},
                        abstol = {abstol}, reltol = {reltol}, dt = {dt},
                        trajectories = {_trajectories},
                        save_everystep = {str(save_everystep).lower()},
                        saveat = {_saveat}, save_idxs = {_save_idxs},
                        save_start = {str(save_start).lower()}
                    );
        """
        )
