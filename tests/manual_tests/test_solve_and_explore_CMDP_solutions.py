from abc import abstractmethod, ABC
from unittest import TestCase

from src.concrete_processes.ecas_examples.dct_example import ForbiddenFloraDCTApprenticeshipCAG
from src.concrete_processes.ecas_examples.pfd_example import FlowerFieldPrimaFacieDuties
from src.concrete_processes.maze_cmdp import RoseMazeCMDP
from src.concrete_processes.rose_garden_cag import RoseGarden
from src.concrete_processes.simplest_cag import SimplestCAG
from src.formalisms.finite_processes import FiniteCMDP
from src.policy_analysis import explore_CMDP_solution_with_trajectories
from src.reductions.cag_to_bcmdp import MatrixCAGtoBCMDP
from src.solvers.linear_programming.cplex_dual_cmdp_solver import solve_CMDP


class TestCMDPSolver(ABC):
    @abstractmethod
    def get_cmdp(self) -> FiniteCMDP:
        raise NotImplementedError

    def setUp(self):
        self.cmdp = self.get_cmdp()

    def test_solve(self):
        policy, solution_details = solve_CMDP(self.cmdp)
        self.explore_solution(policy, solution_details)

    def explore_solution(self, policy, solution_details):
        explore_CMDP_solution_with_trajectories(policy, self.cmdp)


class TestSolveSimpleCAG(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        return MatrixCAGtoBCMDP(SimplestCAG())


class TestSolveRoseMazeCMDP(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        return RoseMazeCMDP()


class TestSolveRoseGarden(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        return MatrixCAGtoBCMDP(RoseGarden())


class TestSolveDCTFlora(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        cag = ForbiddenFloraDCTApprenticeshipCAG(grid_size="small")
        return MatrixCAGtoBCMDP(cag)


class TestSolvePFDFlowers(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        cag = FlowerFieldPrimaFacieDuties(grid_size="small")
        return MatrixCAGtoBCMDP(cag)
