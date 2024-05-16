from typing import Iterable, Sequence

import networkx as nx
import numpy as np
from sympy import Symbol, conjugate, diff, nsimplify, simplify, solve, symbols
from sympy.functions.elementary.exponential import exp as symb_exp
from sympy.matrices import diag, zeros


class Hamiltonian:
    """
    Class for setting up the hamiltonian for a ODE system of Bloch equations.
    """

    def __init__(self, levels: None = None, energies: None | Sequence[Symbol] = None):
        """ """
        if levels is None and energies is None:
            raise ValueError("Supply either the energies or number of levels.")
        elif energies is not None:
            self.energies = energies
            self.levels = len(energies)
            self.hamiltonian = zeros(self.levels, self.levels) + diag(*energies)
        elif levels is not None:
            self.levels = levels
            self.energies = symbols(f"E0:{levels}", real=True)
            self.hamiltonian = zeros(self.levels, self.levels)

        # defining the couplings and rabi rate matrices
        # couplings contains the frequencies between levels
        # rabi contains the rabi rates of the couplings between levels
        self.couplings = zeros(self.levels, self.levels)
        self.rabi = zeros(self.levels, self.levels)

        # level detunings list
        self.detunings = []

        # frequencies list
        self.frequencies = []

        # energies list

        # by default no zero energy defined
        self.zero_energy = None

        self.transformed = None

    def define_zero(self, zero_energy):
        """
        Defining the zero level energy

        Parameters:
        zero_energy : state Symbol() which to set to zero
        """
        if zero_energy not in self.energies.free_symbols:
            raise AssertionError("Specified energy not in any of the energy levels.")
        self.zero_energy = zero_energy

        if self.transformed:
            self.transformed = self.transformed.subs(zero_energy, 0)

    def add_energies(self, energies):
        """
        Adding energy levels to the Hamiltonian

        Parameters:
        energies : list of state Symbol() which define the energy levels
        """
        if not len(energies) == self.levels:
            raise AssertionError("Not all energies specified.")

        # energies are on the diagonal of the Hamiltonian matrix
        for idx, energy in enumerate(energies):
            self.hamiltonian[idx, idx] = energy
        self.energies = self.hamiltonian.diagonal()

    def add_coupling(self, initial, final, rabi, omega):
        """
        Add a coupling between two states

        Parameters:
        initial_state  : initial coupled state
        final_state    : final coupled state
        rabi_rate      : rabi rate of coupling between initial and final, Symbol
        omega          : requency of coupling between initial and final, Symbol
        """
        if (initial > self.hamiltonian.shape[0]) or (final > self.hamiltonian.shape[0]):
            raise AssertionError("Specified state exceeds size of Hamiltonian")

        # setting the frequency and rabi rate of the coupling to the symbolic
        # matrices
        self.couplings[initial, final] = omega
        self.rabi[initial, final] = rabi

        # adding the coupling frequency to the frequencies list if not already
        # present
        if omega not in self.frequencies:
            self.frequencies.append(omega)

        # adding the appropriote terms to the symbolic Hamiltonian matrix
        t = Symbol("t", real=True)
        self.hamiltonian[initial, final] -= rabi / 2 * symb_exp(1j * omega * t)
        self.hamiltonian[final, initial] -= (
            conjugate(rabi) / 2 * symb_exp(-1j * omega * t)
        )

    def add_manifold_coupling(self, initial_manifold, final_manifold, rabis, omega):
        if isinstance(rabis, Iterable):
            for idi, idf, r in zip(initial_manifold, final_manifold, rabis):
                self.add_coupling(idi, idf, r, omega)
        else:
            for idi, idf in zip(initial_manifold, final_manifold):
                self.add_coupling(idi, idf, rabis, omega)

    def define_state_detuning(self, initial, final, detuning):
        """
        Adding a state detuning, requires that self.transformed is defined

        Paramters:
        initial_state : initial coupled state
        final_state   : final coupled state
        detuning      : state detuning, Symbol
        """
        if detuning in self.detunings:
            raise AssertionError("Detuning already defined.")

        # check if the coupling for which the state detuning is requested exists
        if (self.couplings[initial, final] == 0) and (
            self.couplings[final, initial] == 0
        ):
            raise AssertionError("No coupling between states")

        # couplings are defined as initial->final; grab non-zero frequency from
        # initial->final or final->initial (accounts for user error in
        # specifying the detuning initial and final state)
        w = self.couplings[initial, final]
        if w == 0:
            w = self.couplings[final, initial]

        # adding the detuning the the transformed matrix
        self.transformed = self.transformed.subs(
            w, self.energies[final] - self.energies[initial] - detuning
        )

        # append the detuning to the detunings list
        self.detunings.append(
            [
                w,
                self.hamiltonian[final, final],
                self.hamiltonian[initial, initial],
                detuning,
            ]
        )

    def define_energy_detuning(self, initial_energy, final_energy, detuning, omega):
        """ """
        if detuning in self.detunings:
            raise AssertionError("Detuning already defined.")

        if omega not in self.frequencies:
            raise AssertionError("Coupling does not exist")

        self.transformed = self.transformed.subs(
            omega, final_energy - initial_energy - detuning
        )

        if self.zero_energy:
            self.transformed = self.transformed.subs(self.zero_energy, 0)

        self.detunings.append([initial_energy, final_energy, detuning, omega])

    def eqn_transform(self):
        """
        Calculate the rotational wave approximation by solving a system of
        equations, only usable if the number of couplings does not exceed the
        number of states
        """
        assert (
            np.nonzero(self.couplings)[0].size < self.levels
        ), "Number of couplings can't match or exceed the number of levels."

        A = symbols(f"a0:{self.levels}")

        Eqns = []
        for i in range(len(A)):
            for j in range(len(A)):
                if self.couplings[i, j] != 0:
                    Eqns.append(self.couplings[i, j] - (A[i] - A[j]))

        sol = solve(Eqns, A)

        assert len(sol) != 0, "Failed to calculate the rotational wave approximation."

        free_params = [value for value in A if value not in list(sol.keys())]

        for free_param in free_params:
            for key, val in sol.items():
                sol[key] = val.subs(free_param, 0)

        T = zeros(*self.hamiltonian.shape)
        for i in range(self.hamiltonian.shape[0]):
            try:
                T[i, i] = symb_exp(1j * sol[Symbol(f"a{i}")] * Symbol("t", real=True))
            except KeyError:
                T[i, i] = 1
        self.T = T

        self.transformed = T.adjoint() @ self.hamiltonian @ T - 1j * T.adjoint() @ diff(
            T, Symbol("t", real=True)
        )
        self.transformed = nsimplify(simplify(self.transformed))

    def find_dark_states(self, ground_states, excited_states):
        """
        Find the dark states of a system by ...

        Parameters:
        ground_states   : list with indices of the ground states
        excited_states  : list with indices of the excited states

        Returns:
        bright_states   : Matrix with bright states
        dark_states     : Matrix with excited states
        """
        if any(ground_state in excited_states for ground_state in ground_states):
            raise AssertionError("State can not be both ground and excited.")

        M = self.transformed[ground_states, excited_states].adjoint().as_mutable()
        for idx in range(len(excited_states)):
            M[idx, :] /= M[idx, :].norm()
        bright_states = M
        dark_states = M.nullspace()[0]
        return bright_states, dark_states
