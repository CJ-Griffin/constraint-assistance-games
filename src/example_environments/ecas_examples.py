from typing import Set, Tuple

import numpy as np

from src.formalisms.appr_grid_cag import MirrorApprentishipCAG
from src.formalisms.distributions import UniformDiscreteDistribution
from src.formalisms.ecas_cags import DivineCommandTheoryCAG

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


class ReallySimpleDCTApprenticeshipCAG(DivineCommandTheoryCAG, MirrorApprentishipCAG):
    K: int = 1

    def __init__(self, grid_array: np.array = _TINY_GRID, include_empty_forbidden_set: bool = True):
        MirrorApprentishipCAG.__init__(
            self,
            height=grid_array.shape[0],
            width=grid_array.shape[1],
            start=(list(self.find_matching_indeces(grid_array, "0")))[0],
            sinks=self.find_matching_indeces(grid_array, "*"),
            goal_reward=1.0,
            gamma=0.9
        )

        sets_to_include = set(frozenset()) if include_empty_forbidden_set else set()
        for char in ["R", "D", "L"]:
            char_indeces = self.find_matching_indeces(grid_array, char)
            if len(char_indeces) > 0:
                corresponding_states = self.get_corresponding_states(char_indeces)
                sets_to_include.add(corresponding_states)

        poss_Fs = frozenset(sets_to_include)

        DivineCommandTheoryCAG.__init__(self, poss_Fs)
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


class ForbiddenFloraDCTApprenticeshipCAG(DivineCommandTheoryCAG, MirrorApprentishipCAG):
    K: int = 1

    def __init__(self):
        grid_array = _SMALL_GRID

        MirrorApprentishipCAG.__init__(
            self,
            height=grid_array.shape[0],
            width=grid_array.shape[1],
            start=(list(self.find_matching_indeces(grid_array, "0")))[0],
            sinks=self.find_matching_indeces(grid_array, "*"),
            goal_reward=1.0,
            gamma=0.9
        )
        roses = self.find_matching_indeces(grid_array, "R")
        lilies = self.find_matching_indeces(grid_array, "L")
        dasies = self.find_matching_indeces(grid_array, "D")

        poss_Fs = frozenset((
            self.get_corresponding_states(roses),
            self.get_corresponding_states(lilies),
            self.get_corresponding_states(dasies)
        ))

        DivineCommandTheoryCAG.__init__(self, poss_Fs)
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
