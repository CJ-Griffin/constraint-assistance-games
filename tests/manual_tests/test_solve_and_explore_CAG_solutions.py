import cProfile
from abc import abstractmethod, ABC
from pstats import Stats
from unittest import TestCase

from src.concrete_decision_processes.ecas_examples.dct_example import ForbiddenFloraDCTCoop, DCTRoseGardenCoop, \
    DCTRoseGardenAppr
from src.concrete_decision_processes.ecas_examples.pfd_example import FlowerFieldPFDCoop, SimplestFlowerFieldPFDCoop
from src.concrete_decision_processes.rose_garden_cags import RoseGarden, CoopRoseGarden, SimplestCAG
from src.formalisms.finite_processes import FiniteCAG
from src.solution_methods.solvers import get_mixed_solution_to_FiniteCAG
from src.utils.policy_analysis import explore_CAG_policy_with_env_wrapper


class TestCAGSolver(ABC):
    should_profile = False
    should_force_deterministic_cmdp_policy: bool = False

    @abstractmethod
    def get_cag(self) -> FiniteCAG:
        raise NotImplementedError

    def setUp(self):
        """init each test"""
        if self.should_profile:
            self.pr = cProfile.Profile()
            self.pr.enable()
        self.cag = self.get_cag()

    def tearDown(self):
        """finish any test"""
        if self.should_profile:
            p = Stats(self.pr)
            p.strip_dirs()
            p.sort_stats('cumtime')
            p.print_stats()

    def test_solve_and_convert(self):
        mixed_cag_policy = get_mixed_solution_to_FiniteCAG(self.cag)
        for cag_policy in mixed_cag_policy.support():
            print("P={mixed_cag_policy.get_probability(cag_policy)}")
            explore_CAG_policy_with_env_wrapper(cag_policy, self.cag, should_write_to_html=True)


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
        return FlowerFieldPFDCoop()


class TestSolveDCTRoseGardenCoop(TestCAGSolver, TestCase):
    def get_cag(self) -> FiniteCAG:
        return DCTRoseGardenCoop()


class TestDCTRoseGardenAppr(TestCAGSolver, TestCase):
    def get_cag(self) -> FiniteCAG:
        return DCTRoseGardenAppr()


class TestSimplestFlowerFieldPFDCoop(TestCAGSolver, TestCase):
    should_force_deterministic_cmdp_policy = True

    def get_cag(self) -> FiniteCAG:
        return SimplestFlowerFieldPFDCoop()
