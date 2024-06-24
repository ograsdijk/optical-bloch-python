from sympy import Function, Matrix, Rational, Symbol, conjugate, simplify, sqrt
from sympy.matrices import zeros

from .utils.math import anti_commute, commute


class Dissipator:
    """
    Class for setting up the dissipator for the optical Bloch equations.
    """

    def __init__(self, levels):
        """

        Parameters:
        levels  : number of levels in the optical Bloch equations system
        """
        self.density_matrix = zeros(levels, levels)
        self.dissipator = zeros(levels, levels)
        self.branching = zeros(levels, levels)
        self.decay_rate = zeros(levels, 1)
        self.levels = levels

        self.generate_density_matrix()

    def generate_density_matrix(self):
        """
        Generate the symbolic density matrix
        """
        t = Symbol("t", real=True)
        self.density_matrix = Matrix(
            self.levels,
            self.levels,
            lambda i, j: Function(f"ρ{i}{j}")(t)
            if j > i - 1
            else conjugate(Function(f"ρ{j}{i}")(t)),
        )

    def add_decay(self, initial_state, final_state, gamma):
        """
        Add a decay to the dissipator matrix between two states, set by
        initial_state and final_state.

        Parameters:
        initial_state   : initial coupled state
        final_state     : final coupled state
        gamma           : decay rate from initial_state to final_state
        """
        if (initial_state >= self.levels) or (final_state >= self.levels):
            raise AssertionError("Specified state exceeds number of levels.")
        if initial_state == final_state:
            raise AssertionError("State cannot decay into itself.")

        # adding the decay to the dissipator
        G = zeros(self.levels, self.levels)
        G[final_state, initial_state] = sqrt(gamma)
        self.dissipator -= (
            Rational(1 / 2) * anti_commute(G.T @ G, self.density_matrix)
            - G @ self.density_matrix @ G.T
        )
        self.dissipator = simplify(self.dissipator)

        # update decay rates
        decay_rate_old = self.decay_rate[initial_state]
        decay_rate_new = decay_rate_old + gamma
        self.decay_rate[initial_state] += gamma

        # add the decay to the branching ratio matrix
        for i in range(self.levels):
            if i == final_state:
                self.branching[initial_state, i] = (
                    self.branching[initial_state, i] * decay_rate_old + gamma
                ) / decay_rate_new
            else:
                self.branching[initial_state, i] = (
                    self.branching[initial_state, i] * decay_rate_old
                ) / decay_rate_new
                self.branching[initial_state, i] = (
                    self.branching[initial_state, i] * decay_rate_old
                ) / decay_rate_new
