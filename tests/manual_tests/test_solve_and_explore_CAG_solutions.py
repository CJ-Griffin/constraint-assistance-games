import cProfile
from abc import abstractmethod, ABC
from pstats import Stats
from unittest import TestCase

from src.concrete_processes.ecas_examples.dct_example import ForbiddenFloraDCTCoop
from src.concrete_processes.ecas_examples.pfd_example import FlowerFieldPFDCoop
from src.concrete_processes.rose_garden_cags import RoseGarden, CoopRoseGarden, SimplestCAG
from src.formalisms.finite_processes import FiniteCAG
from src.policy_analysis import explore_CAG_policy_with_env_wrapper
from src.solvers.linear_programming.cplex_dual_cmdp_solver import solve_CAG


class TestCAGSolver(ABC):
    @abstractmethod
    def get_cag(self) -> FiniteCAG:
        raise NotImplementedError

    def setUp(self):
        """init each test"""
        self.pr = cProfile.Profile()
        self.pr.enable()
        self.cag = self.get_cag()

    def tearDown(self):
        """finish any test"""
        p = Stats(self.pr)
        p.strip_dirs()
        p.sort_stats('cumtime')
        p.print_stats()

    def test_solve_and_convert(self):
        cag_policy, solution_details = solve_CAG(self.cag)
        explore_CAG_policy_with_env_wrapper(cag_policy, self.cag, should_render=True)


class TestSolveSimpleCAG(TestCAGSolver, TestCase):
    def get_cag(self) -> FiniteCAG:
        return SimplestCAG()


class TestSolveRoseGarden(TestCAGSolver, TestCase):
    def get_cag(self) -> FiniteCAG:
        return RoseGarden()


class TestSolveRoseGardenStoch(TestCAGSolver, TestCase):
    def get_cag(self) -> FiniteCAG:
        return RoseGarden(budget=0.314)


class TestCooperativeCAG(TestCAGSolver, TestCase):
    def get_cag(self) -> FiniteCAG:
        return CoopRoseGarden()


class TestSolveDCTFlora(TestCAGSolver, TestCase):
    def get_cag(self) -> FiniteCAG:
        return ForbiddenFloraDCTCoop(grid_size="medium")


class TestSolveFlowerFieldPrimaFacieDuties(TestCAGSolver, TestCase):
    def get_cag(self) -> FiniteCAG:
        return FlowerFieldPFDCoop(grid_name="small")
