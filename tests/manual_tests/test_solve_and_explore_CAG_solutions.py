from abc import abstractmethod, ABC
from unittest import TestCase

from src.concrete_processes.ecas_examples.dct_example import ForbiddenFloraDCTApprenticeshipCAG
from src.concrete_processes.ecas_examples.pfd_example import FlowerFieldPrimaFacieDuties
from src.concrete_processes.rose_garden_cag import RoseGarden
from src.concrete_processes.simplest_cag import SimplestCAG
from src.formalisms.finite_processes import FiniteCAG
from src.policy_analysis import explore_CAG_policy_with_env_wrapper
from src.solvers.linear_programming.cplex_dual_cmdp_solver import solve_CAG


class TestCMDPSolver(ABC):
    @abstractmethod
    def get_cag(self) -> FiniteCAG:
        raise NotImplementedError

    def setUp(self):
        self.cag = self.get_cag()

    def test_solve_and_convert(self):
        cag_policy, solution_details = solve_CAG(self.cag)
        explore_CAG_policy_with_env_wrapper(cag_policy, self.cag, should_render=True)


class TestSolveSimpleCAG(TestCMDPSolver, TestCase):
    def get_cag(self) -> FiniteCAG:
        return SimplestCAG()


class TestSolveRoseGarden(TestCMDPSolver, TestCase):
    def get_cag(self) -> FiniteCAG:
        return RoseGarden()


class TestSolveRoseGardenStoch(TestCMDPSolver, TestCase):
    def get_cag(self) -> FiniteCAG:
        return RoseGarden(budget=0.314)


class TestSolveDCTFlora(TestCMDPSolver, TestCase):
    def get_cag(self) -> FiniteCAG:
        return ForbiddenFloraDCTApprenticeshipCAG(grid_size="medium")


class TestSolveFlowerFieldPrimaFacieDuties(TestCMDPSolver, TestCase):
    def get_cag(self) -> FiniteCAG:
        return FlowerFieldPrimaFacieDuties(grid_size="small")
