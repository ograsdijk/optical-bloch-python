from typing import Sequence

from sympy import symbols

from ..hamiltonian import Hamiltonian


def hamiltonian_setup(levels: int, couplings: Sequence[tuple[int, int]]) -> Hamiltonian:
    ham = Hamiltonian(levels=levels)
    omegas = symbols(f"ω0:{len(couplings)}", real=True)
    rabis = symbols(f"Ω0:{len(couplings)}")
    deltas = symbols(f"δ0:{len(couplings)}", real=True)
    for (idg, ide), omega, rabi in zip(couplings, omegas, rabis):
        ham.add_coupling(idg, ide, rabi, omega)
    ham.eqn_transform()
    for (idg, ide), delta in zip(couplings, deltas):
        ham.define_state_detuning(idg, ide, delta)
    ham.remove_common_energy()
    return ham
