from typing import Set, Tuple

import numpy as np

from src.concrete_processes.appr_grid_cag import MirrorApprentishipCAG
from src.formalisms.distributions import UniformDiscreteDistribution
from src.formalisms.ecas_cags import DivineCommandTheoryCAG, DCTEthicalContext

_TINY_GRID = np.array([
    ['0', 'R', '*'],
    [' ', ' ', ' ']
])

_SMALL_GRID = np.array([
    ['0', 'R', 'D', '*'],
    [' ', 'R', 'L', 'L'],
    [' ', ' ', ' ', ' ']
])

_MEDIUM_GRID = np.array([
    ['0', 'R', 'D', ' ', ' ', '*'],
    [' ', 'R', 'D', ' ', 'D', 'D'],
    [' ', 'R', 'L', ' ', ' ', 'L'],
    [' ', 'R', 'L', ' ', ' ', ' '],
    [' ', 'R', 'L', 'L', 'L', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ']
])


class ForbiddenFloraDCTApprenticeshipCAG(DivineCommandTheoryCAG, MirrorApprentishipCAG):
    K: int = 1

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
        rose_states = self.get_corresponding_states(self.find_matching_indeces(grid_array, "R"))
        lily_states = self.get_corresponding_states(self.find_matching_indeces(grid_array, "L"))
        daisy_states = self.get_corresponding_states(self.find_matching_indeces(grid_array, "D"))

        ecs = frozenset([
            DCTEthicalContext(forbidden_states=rose_states, nickname="roses"),
            DCTEthicalContext(forbidden_states=lily_states, nickname="lilies"),
            DCTEthicalContext(forbidden_states=daisy_states, nickname="daisies")
        ])

        DivineCommandTheoryCAG.__init__(self, ecs)
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