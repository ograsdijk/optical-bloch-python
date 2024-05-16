from itertools import product
from numbers import Number

import sympy as smp


def generate_code(
    equation_rhs: smp.matrices.MutableDenseMatrix,
    density_matrix: smp.matrices.MutableDenseMatrix,
    replacements: list[tuple[smp.Symbol, Number]] = [],
    conjugate_symbol: str = "conj",
    complex_symbol: str = "1.0im",
    array_start_index: int = 0,
) -> str:
    assert (
        equation_rhs.shape == density_matrix.shape
    ), "equations and density matrix required to be the same shape"

    eqns = smp.nsimplify(smp.simplify(equation_rhs.subs(replacements)))
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


def generate_lindblad_function_julia(
    equation_rhs: smp.matrices.MutableDenseMatrix,
    density_matrix: smp.matrices.MutableDenseMatrix,
    replacements: list[tuple[smp.Symbol, Number]] = [],
) -> tuple[str, tuple[str]]:
    code_lines = generate_code(
        equation_rhs,
        density_matrix,
        replacements=[],
        conjugate_symbol="conj",
        complex_symbol="1.0im",
        array_start_index=1,
    )

    free_parameters = tuple(
        [
            str(par)
            for par in equation_rhs.subs(replacements).free_symbols
            if str(par) != "t"
        ]
    )

    function = """function lindblad!(du, u, p, t)
    \t@inbounds begin
    """

    for idp, parameter in enumerate(free_parameters):
        function += f"\t\t{parameter} = p[{idp+1}]\n"
    for code_line in code_lines:
        function += f"\t\t{code_line}\n"
    function += "\tend\n"
    function += "\tnothing\n"
    function += "end"
    return function, free_parameters
