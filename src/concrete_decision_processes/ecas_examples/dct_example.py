import random
from typing import Set, Tuple

import numpy as np

from src.formalisms.distributions import UniformDiscreteDistribution
from src.formalisms.ecas_cags import DivineCommandTheoryCAG, DCTEthicalContext
from src.abstract_gridworlds.grid_world_cag import CoordinationStaticGridCAG, ApprenticeshipStaticGridCAG

_TINY_GRID = np.array([
    ['h', ' ', 'R', ' ', '*'],
    ['r', ' ', 'L', ' ', '*'],
])

_SMALL_GRID = np.array([
    ['h', 'R', ' ', '*'],
    [' ', 'D', ' ', ' '],
    ['r', 'L', ' ', '*']
])

_MEDIUM_GRID = np.array([
    ['h', 'R', ' ', ' ', ' ', ' '],
    [' ', 'D', ' ', ' ', ' ', ' '],
    ['#', '#', '#', '#', 'D', 'L'],
    ['r', 'L', ' ', ' ', ' ', ' '],
    [' ', 'D', ' ', ' ', '*', '*'],
])

_LARGE_GRID = np.array([
    ['h', 'R', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', 'D', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', 'L', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ['#', '#', '#', '#', '#', '#', '#', 'D', 'L'],
    ['r', 'L', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', 'R', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', 'D', ' ', ' ', ' ', ' ', ' ', '*', '*'],
])

_LARGE_GRID_SPLIT = np.array([
    ['h', 'D', ' ', 'R', ' ', 'R', ' ', 'R'],
    [' ', 'R', ' ', 'L', ' ', 'L', ' ', 'L'],
    [' ', 'L', ' ', 'R', ' ', 'D', ' ', 'R'],
    [' ', 'R', ' ', 'D', ' ', 'L', ' ', '*'],
    ['#', '#', '#', '#', '#', '#', '#', '#'],
    ['r', 'R', ' ', 'R', ' ', 'R', ' ', 'D'],
    [' ', 'D', ' ', 'D', ' ', 'D', ' ', 'L'],
    [' ', 'R', ' ', 'L', ' ', 'D', ' ', 'D'],
    [' ', 'L', ' ', 'D', ' ', 'L', ' ', '*']
])

# Generated using the function below, + some modifications
_XL_GRID_SPLIT = np.array([
    ['h', 'D', ' ', 'L', ' ', 'L', ' ', 'R', ' ', 'R', ' ', 'D', ' ', 'L', ' ', 'R', ' ', 'L', ' ', 'R'],
    [' ', 'D', ' ', 'D', ' ', 'L', ' ', 'R', ' ', 'L', ' ', 'D', ' ', 'L', ' ', 'L', ' ', 'D', ' ', 'R'],
    [' ', 'R', ' ', 'R', ' ', 'R', ' ', 'L', ' ', 'D', ' ', 'D', ' ', 'L', ' ', 'L', ' ', 'D', ' ', 'D'],
    [' ', 'L', ' ', 'R', ' ', 'D', ' ', 'L', ' ', 'L', ' ', 'L', ' ', 'R', ' ', 'R', ' ', 'L', ' ', 'D'],
    [' ', 'L', ' ', 'D', ' ', 'R', ' ', 'L', ' ', 'L', ' ', 'L', ' ', 'D', ' ', 'D', ' ', 'D', ' ', 'L'],
    [' ', 'R', ' ', 'R', ' ', 'L', ' ', 'R', ' ', 'R', ' ', 'R', ' ', 'L', ' ', 'R', ' ', 'R', ' ', 'D'],
    [' ', 'R', ' ', 'L', ' ', 'L', ' ', 'D', ' ', 'D', ' ', 'L', ' ', 'D', ' ', 'D', ' ', 'L', ' ', 'L'],
    [' ', 'L', ' ', 'L', ' ', 'D', ' ', 'D', ' ', 'D', ' ', 'L', ' ', 'R', ' ', 'L', ' ', 'L', ' ', 'L'],
    [' ', 'D', ' ', 'R', ' ', 'R', ' ', 'R', ' ', 'L', ' ', 'R', ' ', 'D', ' ', 'D', ' ', 'L', ' ', 'L'],
    [' ', 'R', ' ', 'R', ' ', 'D', ' ', 'L', ' ', 'R', ' ', 'L', ' ', 'L', ' ', 'L', ' ', 'L', ' ', '*'],
    ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
    ['r', 'R', ' ', 'D', ' ', 'D', ' ', 'R', ' ', 'R', ' ', 'L', ' ', 'R', ' ', 'D', ' ', 'L', ' ', 'R'],
    [' ', 'R', ' ', 'D', ' ', 'D', ' ', 'L', ' ', 'L', ' ', 'L', ' ', 'D', ' ', 'L', ' ', 'R', ' ', 'L'],
    [' ', 'L', ' ', 'D', ' ', 'D', ' ', 'D', ' ', 'D', ' ', 'R', ' ', 'L', ' ', 'D', ' ', 'D', ' ', 'R'],
    [' ', 'D', ' ', 'R', ' ', 'L', ' ', 'R', ' ', 'L', ' ', 'D', ' ', 'D', ' ', 'L', ' ', 'R', ' ', 'L'],
    [' ', 'D', ' ', 'D', ' ', 'D', ' ', 'L', ' ', 'D', ' ', 'L', ' ', 'R', ' ', 'D', ' ', 'L', ' ', 'R'],
    [' ', 'R', ' ', 'R', ' ', 'L', ' ', 'R', ' ', 'L', ' ', 'L', ' ', 'R', ' ', 'R', ' ', 'R', ' ', 'L'],
    [' ', 'L', ' ', 'R', ' ', 'R', ' ', 'L', ' ', 'R', ' ', 'R', ' ', 'D', ' ', 'R', ' ', 'D', ' ', 'D'],
    [' ', 'R', ' ', 'D', ' ', 'R', ' ', 'R', ' ', 'R', ' ', 'R', ' ', 'R', ' ', 'L', ' ', 'D', ' ', 'R'],
    [' ', 'L', ' ', 'L', ' ', 'D', ' ', 'D', ' ', 'L', ' ', 'L', ' ', 'L', ' ', 'D', ' ', 'R', ' ', 'L'],
    [' ', 'D', ' ', 'R', ' ', 'L', ' ', 'R', ' ', 'L', ' ', 'L', ' ', 'R', ' ', 'D', ' ', 'L', ' ', '*']
])


def _generate_dct_split_grid(
        width=20,
        indiv_height=10

):
    grid = np.zeros((indiv_height * 2 + 1, width), dtype=str)
    grid[:, :] = ' '
    grid[indiv_height, :] = '#'
    grid[0, 0] = 'h'
    grid[indiv_height + 1, 0] = 'r'
    grid[indiv_height - 1, -1] = '*'
    grid[-1, -1] = '*'
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == ' ' and j % 2 == 1:
                grid[i, j] = random.choice(['R', 'D', 'L'])
    return grid


_XXL_GRID_SPLIT = _generate_dct_split_grid(30, 15)


class ForbiddenFloraDCTCoop(DivineCommandTheoryCAG, CoordinationStaticGridCAG):

    def __init__(self, grid_size: str = "tiny", size_of_Theta: int = None):
        if grid_size == "tiny":
            grid_array = _TINY_GRID
        elif grid_size == "small":
            grid_array = _SMALL_GRID
        elif grid_size == "medium":
            grid_array = _MEDIUM_GRID
        elif grid_size == "large":
            grid_array = _LARGE_GRID_SPLIT
        else:
            raise ValueError(grid_size)

        if size_of_Theta is None:
            size_of_Theta = 2 if grid_size == "tiny" else 3
        else:
            assert size_of_Theta <= 3

        CoordinationStaticGridCAG.__init__(
            self,
            grid=grid_array
        )
        rose_states = self.get_corresponding_states(self.find_matching_indeces(grid_array, "R"))
        lily_states = self.get_corresponding_states(self.find_matching_indeces(grid_array, "L"))
        daisy_states = self.get_corresponding_states(self.find_matching_indeces(grid_array, "D"))

        ecs = [
            DCTEthicalContext(forbidden_states=rose_states, nickname="roses"),
            DCTEthicalContext(forbidden_states=lily_states, nickname="lilies"),
            DCTEthicalContext(forbidden_states=daisy_states, nickname="daisies")
        ]
        ecs_set = frozenset(ecs[:size_of_Theta])
        print(len(ecs_set))

        DivineCommandTheoryCAG.__init__(self, ecs_set)
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


class SmallForbiddenFloraDCT(ForbiddenFloraDCTCoop):
    def __init__(self):
        super().__init__(grid_size="small")


class MediumForbiddenFloraDCT(ForbiddenFloraDCTCoop):
    def __init__(self):
        super().__init__(grid_size="medium")


class MediumForbiddenFloraDCTTwoECs(ForbiddenFloraDCTCoop):
    def __init__(self):
        super().__init__(grid_size="medium", size_of_Theta=2)


class DCTRoseGardenCoop(DivineCommandTheoryCAG, CoordinationStaticGridCAG):
    def __init__(self):
        grid_array = np.array([
            [" ", "*", "#", "*", " "],
            [" ", "R", "#", "R", " "],
            [" ", "h", "#", "r", " "]
        ])

        CoordinationStaticGridCAG.__init__(
            self,
            grid=grid_array
        )
        rose_states = self.get_corresponding_states(self.find_matching_indeces(grid_array, "R"))

        ecs = [
            DCTEthicalContext(forbidden_states=frozenset(), nickname="<Ƒ_∅>"),
            DCTEthicalContext(forbidden_states=rose_states, nickname="<Ƒ_roses>")
        ]

        DivineCommandTheoryCAG.__init__(self, frozenset(ecs))
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


class ForbiddenFloraDCTAppr(DivineCommandTheoryCAG, ApprenticeshipStaticGridCAG):

    def __init__(self, grid_size: str = "tiny", size_of_Theta: int = None):
        if grid_size == "tiny":
            grid_array = _TINY_GRID
        elif grid_size == "small":
            grid_array = _SMALL_GRID
        elif grid_size == "medium":
            grid_array = _MEDIUM_GRID
        elif grid_size == "large":
            grid_array = _LARGE_GRID_SPLIT
        elif grid_size == "extra_large":
            grid_array = _XL_GRID_SPLIT
        elif grid_size == "extra_extra_large":
            grid_array = _XXL_GRID_SPLIT
        else:
            raise ValueError(grid_size)

        if size_of_Theta is None:
            size_of_Theta = 2 if grid_size == "tiny" else 3
        else:
            assert size_of_Theta <= 3

        ApprenticeshipStaticGridCAG.__init__(
            self,
            grid=grid_array
        )
        rose_states = self.get_corresponding_states(self.find_matching_indeces(grid_array, "R"))
        lily_states = self.get_corresponding_states(self.find_matching_indeces(grid_array, "L"))
        daisy_states = self.get_corresponding_states(self.find_matching_indeces(grid_array, "D"))

        ecs = [
            DCTEthicalContext(forbidden_states=rose_states, nickname="roses"),
            DCTEthicalContext(forbidden_states=lily_states, nickname="lilies"),
            DCTEthicalContext(forbidden_states=daisy_states, nickname="daisies")
        ]
        ecs_set = frozenset(ecs[:size_of_Theta])
        print(len(ecs_set))

        DivineCommandTheoryCAG.__init__(self, ecs_set)
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


class DCTRoseGardenAppr(DivineCommandTheoryCAG, ApprenticeshipStaticGridCAG):

    def __init__(self):
        grid_array = np.array([
            [" ", "*", "#", "*", " "],
            [" ", "R", "#", "R", " "],
            [" ", "h", "#", "r", " "]
        ])

        ApprenticeshipStaticGridCAG.__init__(
            self,
            grid=grid_array
        )
        rose_states = self.get_corresponding_states(self.find_matching_indeces(grid_array, "R"))

        ecs = [
            DCTEthicalContext(forbidden_states=frozenset(), nickname="<∅>"),
            DCTEthicalContext(forbidden_states=rose_states, nickname="<Ƒ_Roses>")
        ]

        DivineCommandTheoryCAG.__init__(self, frozenset(ecs))
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
