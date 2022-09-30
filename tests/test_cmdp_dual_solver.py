import unittest

from unittest import TestCase
from src.example_environments.maze_cmdp import RoseMazeCMDP
from src.solvers.linear_programming.cplex_dual_cmdp_solver import solve
import pprint

GRID_WORLD_WIDTH = 5
GRID_WORLD_HEIGHT = 5
WALL_PROBABILITY = 0.2


class TestGridWorldCMDP(TestCase):
    def setUp(self):
        self.pp = pprint.PrettyPrinter(indent=4)
        self.cmdp = RoseMazeCMDP()


class TestGridWorldDualSolution(TestGridWorldCMDP):
    def test_solve(self):
        solution = solve(self.cmdp, 0.99)
        self.pp.pprint(solution)
