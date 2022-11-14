import numpy as np

from src.formalisms.distributions import UniformDiscreteDistribution
from src.formalisms.primitives import ActionPair
from src.gridworlds.grid_world_cag import ApprenticeshipStaticGridCAG, CoordinationStaticGridCAG
from src.gridworlds.grid_world_primitives import StaticGridState, GridAction

"""
# # # # # # # #
# H   # # # # #
# @   # R @ * #
# *   #       #
# # # # # # # #

self.h_A = {A_NORTH, A_SOUTH, A_EAST, A_WEST, A_NOOP}
"""

_HUMAN_GRID = np.array([
    ['0', ' '],
    ['R', ' '],
    ['*', ' ']
])

_ROBOT_GRID = np.array([
    ['0', 'R', '*'],
    [' ', ' ', ' ']
])


class RoseGarden(ApprenticeshipStaticGridCAG):
    Theta = {"imprm", "prm"}
    initial_theta_dist = UniformDiscreteDistribution(Theta)

    def __init__(self, budget: float = 0.0):
        grid = np.array([
            [" ", "*", "#", "*", " "],
            [" ", "R", "#", "R", " "],
            [" ", "h", "#", "r", " "]
        ]
        )
        self.c_tuple = (budget,)
        super().__init__(grid)

    def _inner_C(self, k: int, theta, s: StaticGridState, h_a: GridAction, r_a: GridAction) -> float:
        if theta == "prm" or self.is_sink(s):
            return 0.0
        else:
            next_dist = self.split_T(s, h_a, r_a)
            if next_dist.is_degenerate():
                next_s: StaticGridState = next_dist.sample()
            else:
                raise ValueError

            h_cell_type = self.background_grid[next_s.h_xy[1], next_s.h_xy[0]]
            r_cell_type = self.background_grid[next_s.r_xy[1], next_s.r_xy[0]]

            h_cost = 1.0 if h_cell_type == "R" else 0.0
            r_cost = 1.0 if r_cell_type == "R" else 0.0

            return h_cost + r_cost


class CoopRoseGarden(CoordinationStaticGridCAG):
    Theta = {"imprm", "prm"}
    initial_theta_dist = UniformDiscreteDistribution(Theta)

    def __init__(self, budget: float = 0.0):
        grid = np.array([
            [" ", "*", "#", "*", " "],
            [" ", "R", "#", "R", " "],
            [" ", "h", "#", "r", " "]
        ]
        )
        self.c_tuple = (budget,)
        super().__init__(grid)

    def _inner_C(self, k: int, theta, s: StaticGridState, h_a: GridAction, r_a: GridAction) -> float:
        if theta == "prm" or self.is_sink(s):
            return 0.0
        else:
            next_dist = self.split_T(s, h_a, r_a)
            if next_dist.is_degenerate():
                next_s: StaticGridState = next_dist.sample()
            else:
                raise ValueError

            h_cell_type = self.background_grid[next_s.h_xy[1], next_s.h_xy[0]]
            r_cell_type = self.background_grid[next_s.r_xy[1], next_s.r_xy[0]]

            h_cost = 1.0 if h_cell_type == "R" else 0.0
            r_cost = 1.0 if r_cell_type == "R" else 0.0

            return h_cost + r_cost


"""
=====HUMAN================ROBOT=====
| strt | slow |      | strt | slow |
| (0,0)| (1,0)|      | (0,0)| (1,0)|
--------------- 
| rose | goal |      | rose | goal |
| (0,1)| (1,1)|      | (0,1)| (1,1)|
====================================
slow is represented by ','
gets R=-0.5 for going through the slow state
gets C=3.0 for going through the roses
"""


class SimplestCAG(ApprenticeshipStaticGridCAG):
    Theta = {"imprm", "prm"}
    initial_theta_dist = UniformDiscreteDistribution(Theta)

    def __init__(self, budget: float = 0.0):
        grid = np.array([
            ["*", "R", "#", "R", "*"],
            [",", "h", "#", "r", ","]
        ]
        )
        self.c_tuple = (budget,)
        super().__init__(grid)

    def _inner_R(self, s: StaticGridState, action_pair: ActionPair) -> float:
        base_r = super(SimplestCAG, self)._inner_R(s, action_pair)
        if self.is_sink(s):
            return base_r
        else:
            if self.background_grid[s.h_xy[1], s.h_xy[0]] == ",":
                return base_r - 0.5
            elif self.background_grid[s.r_xy[1], s.r_xy[0]] == ",":
                return base_r - 0.5
            else:
                return base_r

    def _inner_C(self, k: int, theta, s: StaticGridState, h_a: GridAction, r_a: GridAction) -> float:
        if theta == "prm" or self.is_sink(s):
            return 0.0
        else:
            next_dist = self.split_T(s, h_a, r_a)
            if next_dist.is_degenerate():
                next_s: StaticGridState = next_dist.sample()
            else:
                raise ValueError

            h_cell_type = self.background_grid[next_s.h_xy[1], next_s.h_xy[0]]
            r_cell_type = self.background_grid[next_s.r_xy[1], next_s.r_xy[0]]

            h_cost = 1.0 if h_cell_type == "R" else 0.0
            r_cost = 1.0 if r_cell_type == "R" else 0.0

            return h_cost + r_cost
