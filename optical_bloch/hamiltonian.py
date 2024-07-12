from typing import Iterable, Sequence

import numpy as np
from sympy import (
    Matrix,
    Symbol,
    conjugate,
    diff,
    eye,
    nsimplify,
    simplify,
    solve,
    symbols,
)
from sympy.functions.elementary.exponential import exp as symb_exp
from sympy.matrices import diag, zeros


class Hamiltonian:
    """
    Class for setting up the hamiltonian for a ODE system of Bloch equations.
    """

    def __init__(
        self, levels: None | int = None, energies: None | Sequence[Symbol] = None
    ):
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
            self.hamiltonian = zeros(self.levels, self.levels) + diag(*self.energies)

        # defining the couplings and rabi rates
        # couplings contains the frequencies between levels
        # rabis contains the rabi rates of the couplings between levels
        self.couplings: dict[(int, int), Symbol] = {}
        self.rabis: dict[(int, int), Symbol] = {}

        # level detunings list
        self.detunings: list[Symbol] = []

        # frequencies list
        self.frequencies: list[Symbol] = []

        self.transformed = None

    def define_zero(self, zero_energy):
        """
        Defining the zero level energy

        Parameters:
        zero_energy : state Symbol() which to set to zero
        """
        if zero_energy not in self.energies:
            raise AssertionError("Specified energy not in any of the energy levels.")

        if self.transformed:
            self.transformed -= zero_energy * eye(self.levels)

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
        self.rabis[initial, final] = rabi

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

    def add_manifold_coupling(
        self,
        initial_manifold: Sequence[int],
        final_manifold: Sequence[int],
        rabis: Sequence[Symbol] | Symbol,
        omega: Symbol,
    ) -> None:
        if isinstance(rabis, Iterable):
            for idi, idf, r in zip(initial_manifold, final_manifold, rabis):
                self.add_coupling(idi, idf, r, omega)
        else:
            for idi, idf in zip(initial_manifold, final_manifold):
                self.add_coupling(idi, idf, rabis, omega)

    def define_state_detuning(self, initial: int, final: int, detuning: Symbol):
        """
        Adding a state detuning, requires that self.transformed is defined

        Parameters:
        initial_state : initial coupled state
        final_state   : final coupled state
        detuning      : state detuning, Symbol
        """
        if detuning in [d[-1] for d in self.detunings]:
            raise AssertionError("Detuning already defined.")

        # check if the coupling for which the state detuning is requested exists
        if self.couplings.get((initial, final)) is None:
            raise AssertionError("No coupling between states")

        w = self.couplings[(initial, final)]

        # adding the detuning the the transformed matrix
        if initial < final:
            self.transformed = self.transformed.subs(
                w, self.energies[final] - self.energies[initial] - detuning
            )
        elif final < initial:
            self.transformed = self.transformed.subs(
                w, self.energies[initial] - self.energies[final] - detuning
            )

        # append the detuning to the detunings list
        self.detunings.append(
            [
                w,
                self.hamiltonian[initial, initial],
                self.hamiltonian[final, final],
                detuning,
            ]
        )

    def setup_detunings(self) -> None:
        """
        Replace ω with Ef-Ei - δ for each defined coupling
        """
        for idc, ((idg, ide), omega) in enumerate(self.couplings.items()):
            detuning = Symbol(f"δ{idc}", real=True)
            self.define_state_detuning(idg, ide, detuning)

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
                if self.couplings.get((i, j)) is not None:
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

    def remove_common_energy(self) -> None:
        """
        Remove common energy offsets from all diagonal entries of the hamiltonian by
        checking which energies are present in all ground states as defined by the
        couplings dictionary
        """
        # check which symbolic variables occur in all diagonal ground state entries and
        # remove them from the diagonal

        # get ground states entries of the diagonal
        coupling_ground_states = np.unique([key[0] for key in self.couplings.keys()])
        diagonal = Matrix(
            np.array(self.transformed.diagonal())[0, coupling_ground_states]
        )

        for par in diagonal.free_symbols:
            # skip if the parameter is a detuning
            if par in [d[-1] for d in self.detunings]:
                continue

            flag = True
            for val in diagonal:
                if par not in val.free_symbols:
                    flag = False
            if flag:
                self.define_zero(par)

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

    # def shortestCouplingPath(self, graph, initial_state):
    #     """
    #     """
    #     # find the indices of the states which are defined as the zero_energy
    #     indices_zero_energy = np.where(
    #                             np.array(self.energies) == self.zero_energy)[1]
    #     if indices_zero_energy.size == 0:
    #         return 0

    #     # get the first succesful shortest path from the initial state to a
    #     # state with zero_energy
    #     for idx in indices_zero_energy:
    #         shortest_path = nx.algorithms.shortest_path(graph,
    #                     source=initial_state, target = idx, weight = 'weight')
    #         if shortest_path:
    #             break

    #     print(initial_state, shortest_path)

    #     # calculate the total coupling phase between the initial state and final
    #     # state
    #     phase = 0
    #     for j in range(len(shortest_path)-1):
    #         start, stop = shortest_path[j:j+2]
    #         if self.couplings[start,stop] != 0:
    #             phase += self.couplings[start, stop]
    #         else:
    #             phase -= self.couplings[stop, start]
    #     return phase

    # def generalTransform(self, graph):
    #     """
    #     """
    #     if not self.zero_energy:
    #         raise AssertionError(
    #         'Zero energy has to be specified for this transformation method.')

    #     t = Symbol('t', real = True)

    #     T = eye(self.levels)
    #     for i in range(self.levels):
    #         phase = self.shortestCouplingPath(graph, i)
    #         T[i,i] = T[i,i]*symb_exp(1j*phase*t)
    #     T = simplify(T)

    #     self.transformed = T.adjoint()@self.hamiltonian@T \
    #                             -1j*T.adjoint()@diff(T,Symbol('t', real = True))

    #     if self.detunings:
    #         print(self.detunings)
    #         for i in range(len(self.detunings)):
    #             detuning = self.detunings[i]
    #             self.transformed = self.transformed.subs(
    #                         detuning[3], detuning[1] - detuning[0] - detuning[2]
    #                         )

    #     if self.zero_energy:
    #         self.transformed = self.transformed.subs(self.zero_energy, 0)

    #     self.T = T
