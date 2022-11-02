from typing import Set, Tuple

import numpy as np

from src.concrete_processes.appr_grid_cag import MirrorApprentishipCAG
from src.formalisms.distributions import UniformDiscreteDistribution
from src.formalisms.ecas_cags import PrimaFacieDutiesCAG, PFDEthicalContext

_TINY_GRID = np.array([
    ['0', 'R', '*'],
    ['D', 'L', 'D']
])

_SMALL_GRID = np.array([
    ['0', 'R', 'D', '*'],
    ['D', 'R', 'L', 'L'],
    ['L', 'L', 'L', 'D']
])

_MEDIUM_GRID = np.array([
    ['0', 'R', 'D', 'R', 'L', '*'],
    ['D', 'R', 'D', 'R', 'D', 'D'],
    ['D', 'R', 'L', 'L', 'R', 'L'],
    ['L', 'D', 'L', 'D', 'R', 'D'],
    ['L', 'R', 'L', 'L', 'L', 'L'],
    ['D', 'L', 'L', 'R', 'D', 'R']
])


class FlowerFieldPrimaFacieDuties(PrimaFacieDutiesCAG, MirrorApprentishipCAG):
    def __init__(self, grid_size: str = "medium"):
        if grid_size == "tiny":
            grid_array = _TINY_GRID
        elif grid_size == "small":
            grid_array = _SMALL_GRID
        elif grid_size == "medium":
            grid_array = _MEDIUM_GRID
        else:
            raise ValueError(grid_size)

        MirrorApprentishipCAG.__init__(
            self,
            height=grid_array.shape[0],
            width=grid_array.shape[1],
            start=(list(self.find_matching_indeces(grid_array, "0")))[0],
            sinks=self.find_matching_indeces(grid_array, "*"),
            goal_reward=1.0,
            gamma=0.9
        )

        duties = frozenset(("roses", "daisies", "lilies"))

        self.rose_states = self.get_corresponding_states(self.find_matching_indeces(grid_array, "R"))
        self.lily_states = self.get_corresponding_states(self.find_matching_indeces(grid_array, "L"))
        self.daisy_states = self.get_corresponding_states(self.find_matching_indeces(grid_array, "D"))

        heavy_penalty = 10.0
        medium_penalty = 2.0
        light_penalty = 0.5

        def rose_heavy_penalty_function(s_next, duty):
            if duty == "roses" and s_next in self.rose_states:
                return heavy_penalty
            elif duty == "daisies" and s_next in self.daisy_states:
                return medium_penalty
            elif duty == "lilies" and s_next in self.lily_states:
                return light_penalty
            else:
                return 0.0

        def daisy_heavy_penalty_function(s_next, duty):
            if duty == "daisies" and s_next in self.daisy_states:
                return heavy_penalty
            elif duty == "lilies" and s_next in self.lily_states:
                return medium_penalty
            elif duty == "roses" and s_next in self.rose_states:
                return light_penalty
            else:
                return 0.0

        def lily_heavy_penalty_function(s_next, duty):
            if duty == "lilies" and s_next in self.lily_states:
                return heavy_penalty
            elif duty == "roses" and s_next in self.rose_states:
                return medium_penalty
            elif duty == "daisies" and s_next in self.daisy_states:
                return light_penalty
            else:
                return 0.0

        rose_weighted_ec = PFDEthicalContext(duties, rose_heavy_penalty_function, tolerance=10, nickname="R>D>L")
        daisy_weighted_ec = PFDEthicalContext(duties, daisy_heavy_penalty_function, tolerance=10, nickname="D>L>R")
        lily_weighted_ec = PFDEthicalContext(duties, lily_heavy_penalty_function, tolerance=10, nickname="L>R>D")

        ethical_contexts = frozenset([rose_weighted_ec, daisy_weighted_ec, lily_weighted_ec])

        PrimaFacieDutiesCAG.__init__(self, ethical_contexts)
        self.initial_state_theta_dist = UniformDiscreteDistribution({
            (self.s_0, theta) for theta in self.Theta
        })

    @staticmethod
    def find_matching_indeces(grid: np.array, char: str) -> Set[Tuple[int, int]]:
        return set([(pos[1], pos[0]) for pos in np.argwhere(grid == char)])

    def get_corresponding_states(self, forbidden_coords) -> frozenset:
        return frozenset({
            state for state in self.S
            if state.h_xy in forbidden_coords or state.r_xy in forbidden_coords
        })
