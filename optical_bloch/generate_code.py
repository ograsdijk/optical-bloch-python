import re
from itertools import product
from numbers import Number
from pathlib import Path

import networkx as nx
import sympy as smp

from .utils.general import flatten


def generate_code(
    equation_rhs: smp.matrices.MutableDenseMatrix,
    density_matrix: smp.matrices.MutableDenseMatrix,
    replacements: list[tuple[smp.Symbol, Number]] = [],
    conjugate_symbol: str = "conj",
    complex_symbol: str = "1.0im",
    array_start_index: int = 0,
    simplify: bool = True,
) -> str:
    assert (
        equation_rhs.shape == density_matrix.shape
    ), "equations and density matrix required to be the same shape"

    if simplify:
        eqns = smp.nsimplify(smp.simplify(equation_rhs.subs(replacements)))
    else:
        eqns = equation_rhs.subs(replacements)

    shape = equation_rhs.shape
    levels = shape[0]

    code_lines: list[str] = []

    for idx in range(levels):
        for idy in range(idx, levels):
            if eqns[idx, idy] != 0:
                eq = eqns[idx, idy]
                cline = f"{eq}"
                cline = cline.replace("conjugate", conjugate_symbol)
                cline = cline.replace("I*", f"{complex_symbol}*")
                for i, j in product(range(levels), (range(levels))):
                    cline = cline.replace(
                        str(density_matrix[i, j]),
                        f"u[{i+array_start_index},{j+array_start_index}]",
                    )

                # for i in range(levels):
                #     for j in range(i, levels):
                #         cline = cline.replace(
                #             f"u[{i+array_start_index},{j+array_start_index}]",
                #             f"{conjugate_symbol}(u[{j+array_start_index},{i+array_start_index}])",
                #         )

                code_lines.append(
                    f"du[{idx+array_start_index},{idy+array_start_index}] = {cline}"
                )

    return code_lines


def get_compound_parameter_order(
    compound_parameters: list[tuple[smp.Symbol, list[smp.Symbol]]],
) -> list[smp.Symbol]:
    # construct a directed graph
    graph = nx.DiGraph()
    graph.add_nodes_from([k for k, _ in compound_parameters])
    graph.add_node(smp.Symbol("Ω3"))

    # add edges
    for k, v in compound_parameters:
        for vi in v:
            graph.add_edge(vi, k)
    # this returns the edges ordered by order of nodes and node adjacency
    edges = list(graph.edges())
    compound_ordered: list[smp.Symbol] = []
    for i, f in edges:
        if i not in compound_ordered:
            compound_ordered.append(i)
        if f not in compound_ordered:
            compound_ordered.append(f)
    return compound_ordered


def get_parameters_functions(
    equation_rhs: smp.matrices.MutableDenseMatrix,
    replacements: list[tuple[smp.Symbol, Number | smp.Expr | str]],
) -> tuple[
    tuple[smp.Symbol, ...],
    tuple[tuple[smp.Symbol, Number], ...],
    tuple[tuple[smp.Symbol, smp.Expr | str], ...],
    tuple[smp.Function],
]:
    _replacements = []
    for parameter, value in replacements:
        if isinstance(value, str):
            _replacements.append(
                (parameter, smp.parsing.sympy_parser.parse_expr(value))
            )
        else:
            _replacements.append((parameter, value))

    free_parameters = sorted(
        [
            par
            for par in equation_rhs.subs(_replacements).free_symbols
            if str(par) != "t"
        ],
        key=str,
    )

    # grab parameters from compound parameters and put them in free parameters
    if len(_replacements) > 0:
        for parameter, value in _replacements:
            if isinstance(value, smp.Expr):
                if len(value.free_symbols) > 0:
                    for par in value.free_symbols:
                        if (par not in free_parameters) and (str(par) != "t"):
                            free_parameters.append(par)

    # grab non-compound parameters from replacements
    fixed_parameters = [
        (par, val) for par, val in _replacements if not isinstance(val, smp.Expr)
    ]

    # remove any parameters in free parameters that are actually in replacements
    free_parameters = sorted(free_parameters, key=str)
    for par, _ in _replacements:
        if par in free_parameters:
            free_parameters.remove(par)

    # reorder compound parameters such that all are defined before they are called
    compound_parameters = [
        (par, val) for par, val in _replacements if isinstance(val, smp.Expr)
    ]
    compound_parameters_ordered = get_compound_parameter_order(
        [
            (par, [p for p in val.free_symbols if str(p) != "t"])
            for par, val in compound_parameters
        ]
    )

    other_parameters = free_parameters + [p for p, _ in fixed_parameters]
    compound_parameters = [
        (par, dict(compound_parameters)[par])
        for par in compound_parameters_ordered
        if par not in other_parameters
    ]

    functions = flatten(
        [
            [fun.func for fun in expression.atoms(smp.Function)]
            for _, expression in compound_parameters
        ]
    )

    return free_parameters, fixed_parameters, compound_parameters, functions


def defined_julia_functions() -> list[str]:
    julia_functions: list[str] = ["sin", "cos", "exp"]

    with open(Path(__file__).parent / "julia_functions.jl", "r") as f:
        for line in f.readlines():
            if line.strip().startswith("function"):
                match = re.search(r"function\s+(\w+)\(", line.strip())
                if match:
                    julia_functions.append(match.group(1))
    return julia_functions


def generate_lindblad_function_julia(
    equation_rhs: smp.matrices.MutableDenseMatrix,
    density_matrix: smp.matrices.MutableDenseMatrix,
    replacements: list[tuple[smp.Symbol, Number | smp.Expr]] = [],
    function_name: str = "lindblad!",
    simplify: bool = True,
) -> tuple[
    str,
    tuple[smp.Symbol, ...],
    tuple[tuple[smp.Symbol, smp.Expr], ...],
    tuple[tuple[smp.Symbol, Number], ...],
]:
    code_lines = generate_code(
        equation_rhs,
        density_matrix,
        replacements=[],
        conjugate_symbol="conj",
        complex_symbol="1.0im",
        array_start_index=1,
        simplify=simplify,
    )

    free_parameters, fixed_parameters, compound_parameters, functions = (
        get_parameters_functions(equation_rhs, replacements)
    )

    julia_functions = defined_julia_functions()
    for func in functions:
        assert str(func) in julia_functions, f'Function "{func}" not defined in Julia'

    function = f"""function {function_name}(du, u, p, t)
    \t@inbounds begin
    """
    for idp, parameter in enumerate(free_parameters):
        function += f"\t\t{parameter} = p[{idp+1}]\n"
    if len(replacements) > 0:
        for parameter, value in fixed_parameters:
            function += f"\t\t{parameter} = {value}\n"
        for parameter, value in compound_parameters:
            function += f"\t\t{parameter} = {value}\n"
    for code_line in code_lines:
        function += f"\t\t{code_line}\n"
    function += "\tend\n"
    function += "\tnothing\n"
    function += "end"

    return (
        function,
        tuple(free_parameters),
        tuple(compound_parameters),
        tuple(fixed_parameters),
    )
