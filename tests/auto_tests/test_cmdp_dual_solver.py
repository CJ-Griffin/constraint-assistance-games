import pprint
from unittest import TestCase

from src.example_environments.maze_cmdp import RoseMazeCMDP
from src.example_environments.simplest_cag import SimplestCAG
from src.formalisms.cag_to_bcmdp import CAGtoBCMDP
from src.formalisms.cmdp import FiniteCMDP
from src.solvers.linear_programming.cplex_dual_cmdp_solver import solve

GRID_WORLD_WIDTH = 5
GRID_WORLD_HEIGHT = 5
WALL_PROBABILITY = 0.2


class TestCMDPSolver(TestCase):
    cmdp: FiniteCMDP = None
    pp = pprint.PrettyPrinter(indent=4)

    def setUp(self):
        self.cmdp = RoseMazeCMDP()
        self.cmdp.check_matrices()

    def test_solve(self):
        """
        Note these tests are for funding Exceptions and *not* for testing validity of solutions.
        :return:
        """
        _ = solve(self.cmdp)


class TestDualSolveSimpleCAG(TestCMDPSolver):
    def setUp(self):
        self.cag = SimplestCAG()
        self.cmdp = CAGtoBCMDP(self.cag)
        self.cmdp.check_matrices()
