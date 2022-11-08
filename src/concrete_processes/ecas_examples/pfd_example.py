from typing import Set, Tuple

import numpy as np

from src.formalisms.distributions import UniformDiscreteDistribution
from src.formalisms.ecas_cags import PrimaFacieDutiesCAG, PFDEthicalContext
from src.grid_world_cag import CoordinationStaticGridCAG, ApprenticeshipStaticGridCAG2
from src.grid_world_primitives import StaticGridState


def _construct_PFD_ec(rose_penalty: float, lily_penalty: float, daisy_penalty: float, tolerance: float):
    def penalty_function(s_next: StaticGridState, duty: str) -> float:
        h_cell = s_next.get_human_cell()
        r_cell = s_next.get_robot_cell()
        if duty == "roses":
            return rose_penalty * sum([h_cell == "R", r_cell == "R"])
        elif duty == "daisies":
            return daisy_penalty * sum([h_cell == "D", r_cell == "D"])
        elif duty == "lilies":
            return lily_penalty * sum([h_cell == "L", r_cell == "L"])
        else:
            raise ValueError

    return PFDEthicalContext(
        duties=frozenset(("roses", "daisies", "lilies")),
        penalty_function=penalty_function,
        tolerance=tolerance,
        nickname=f"R{rose_penalty};L{lily_penalty};D{daisy_penalty}"
    )


class FlowerFieldPFDCoop(PrimaFacieDutiesCAG, CoordinationStaticGridCAG):
    _tolerance: float = 5.0
    _grid_array: np.array = np.array([
        ['h', ' ', 'R', ' ', '*'],
        ['r', ' ', 'L', 'D', '*']
    ])

    def __init__(self):
        self._grid_array.flags.writeable = False
        CoordinationStaticGridCAG.__init__(
            self,
            grid=self._grid_array
        )

        heavy_penalty = 10.0
        medium_penalty = 2.0
        light_penalty = 0.5

        ethical_contexts = frozenset([
            _construct_PFD_ec(
                rose_penalty=heavy_penalty,
                lily_penalty=medium_penalty,
                daisy_penalty=light_penalty,
                tolerance=self._tolerance
            ),
            _construct_PFD_ec(
                rose_penalty=light_penalty,
                lily_penalty=heavy_penalty,
                daisy_penalty=medium_penalty,
                tolerance=self._tolerance
            ),

            _construct_PFD_ec(
                rose_penalty=medium_penalty,
                lily_penalty=light_penalty,
                daisy_penalty=heavy_penalty,
                tolerance=self._tolerance
            ),
        ])

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


class SmallFlowerFieldPFDCoop(FlowerFieldPFDCoop):
    _tolerance = 10
    _grid_array = np.array([
        ['h', 'R', ' ', '*'],
        ['D', 'R', 'L', 'D'],
        ['D', ' ', ' ', 'L'],
        ['r', 'L', 'R', '*']
    ])


class MediumFlowerFieldPFDCoop(FlowerFieldPFDCoop):
    _tolerance = 10.0
    _grid_array = np.array([
        ['h', 'R', 'D', 'R', 'L', '*'],
        ['D', 'R', 'D', 'R', 'D', 'D'],
        ['D', 'R', 'L', 'L', 'R', 'L'],
        ['L', 'D', 'L', 'D', 'R', 'D'],
        ['L', 'R', 'L', 'L', 'L', 'L'],
        ['r', 'L', 'L', 'R', 'D', '*']
    ])


class ForceStochasticFlowerFieldPFDCoop(PrimaFacieDutiesCAG, CoordinationStaticGridCAG):
    _tolerance = 5.0
    _grid_array = np.array([
        ['h', ' ', 'R', ' ', '*'],
        ['#', '#', '#', '#', '#'],
        ['r', ' ', 'R', ' ', '*']
    ])

    def __init__(self):
        self._grid_array.flags.writeable = False
        CoordinationStaticGridCAG.__init__(
            self,
            grid=self._grid_array
        )

        ethical_contexts = frozenset([
            _construct_PFD_ec(
                rose_penalty=10.0,
                lily_penalty=10.0,
                daisy_penalty=10.0,
                tolerance=self._tolerance
            )
        ])

        PrimaFacieDutiesCAG.__init__(self, ethical_contexts)
        self.initial_state_theta_dist = UniformDiscreteDistribution({
            (self.s_0, theta) for theta in self.Theta
        })


class BreaksReductionFlowerFieldPFDAppr(PrimaFacieDutiesCAG, ApprenticeshipStaticGridCAG2):
    _tolerance = 1.0
    _grid_array = np.array([
        ['*', ' ', ' ', ' ', 'h', 'R', '*'],
        ['#', '#', '#', '#', '#', '#', '#'],
        ['*', ' ', ' ', ' ', 'r', 'R', '*']
    ])

    def __init__(self):
        self._grid_array.flags.writeable = False
        CoordinationStaticGridCAG.__init__(
            self,
            grid=self._grid_array,
            gamma=0.8
        )

        ethical_contexts = frozenset([
            _construct_PFD_ec(
                rose_penalty=10.0,
                lily_penalty=10.0,
                daisy_penalty=10.0,
                tolerance=self._tolerance
            ),
            _construct_PFD_ec(
                rose_penalty=0.0,
                lily_penalty=0.0,
                daisy_penalty=0.0,
                tolerance=self._tolerance
            ),
        ])

        PrimaFacieDutiesCAG.__init__(self, ethical_contexts)
        self.initial_state_theta_dist = UniformDiscreteDistribution({
            (self.s_0, theta) for theta in self.Theta
        })
