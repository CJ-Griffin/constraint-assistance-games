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


class ForbiddenFloraDCTCoop(DivineCommandTheoryCAG, CoordinationStaticGridCAG):

    def __init__(self, grid_size: str = "tiny", size_of_Theta: int = None):
        if grid_size == "tiny":
            grid_array = _TINY_GRID
        elif grid_size == "small":
            grid_array = _SMALL_GRID
        elif grid_size == "medium":
            grid_array = _MEDIUM_GRID
        elif grid_size == "large":
            grid_array = _LARGE_GRID
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
