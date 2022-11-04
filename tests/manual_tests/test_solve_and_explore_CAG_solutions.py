from abc import abstractmethod, ABC
from unittest import TestCase

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
