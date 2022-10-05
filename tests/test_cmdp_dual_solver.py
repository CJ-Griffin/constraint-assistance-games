import unittest

from unittest import TestCase
from src.example_environments.maze_cmdp import RoseMazeCMDP
from src.formalisms.cag_to_bcmdp import CAG_to_BMDP
from src.solvers.linear_programming.cplex_dual_cmdp_solver import solve
from src.example_environments.rose_garden_cag import RoseGarden
import pprint

GRID_WORLD_WIDTH = 5
GRID_WORLD_HEIGHT = 5
WALL_PROBABILITY = 0.2


class TestCMDPSolver(TestCase):
    def setUp(self):
        self.pp = pprint.PrettyPrinter(indent=4)
        self.cmdp = RoseMazeCMDP()

    def test_solve(self):
        solution = solve(self.cmdp, 0.99)
        self.pp.pprint(solution)


class TestCMDPSolverOnCAGReduction(TestCase):
    def setUp(self):
        self.pp = pprint.PrettyPrinter(indent=4)
        self.cag = RoseGarden()
        pickle_path = "cag_to_bmdp.pick"
        self.cmdp = CAG_to_BMDP(self.cag)
        self.cmdp.validate()

    def test_solve(self):
        solution = solve(self.cmdp, 0.99)
        self.pp.pprint(solution)
