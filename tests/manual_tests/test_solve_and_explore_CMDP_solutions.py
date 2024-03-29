import cProfile
from abc import abstractmethod, ABC
from pstats import Stats
from unittest import TestCase

from src.concrete_decision_processes.ecas_examples.dct_example import ForbiddenFloraDCTCoop, DCTRoseGardenAppr
from src.concrete_decision_processes.ecas_examples.pfd_example import FlowerFieldPFDCoop, SmallFlowerFieldPFDCoop, \
    ForceStochasticFlowerFieldPFDCoop, BreaksReductionFlowerFieldPFDAppr, SimplestFlowerFieldPFDCoop
from src.concrete_decision_processes.maze_cmdp import RoseMazeCMDP
from src.concrete_decision_processes.rose_garden_cags import RoseGarden, CoopRoseGarden, SimplestCAG
from src.formalisms.finite_processes import FiniteCMDP
from src.reductions.cag_to_bcmdp import MatrixCAGtoBCMDP
from src.solution_methods.solvers import get_policy_solution_to_FiniteCMDP
from src.utils.policy_analysis import explore_CMDP_solution_with_trajectories


class TestCMDPSolver(ABC):
    should_profile: bool = False
    should_split_stoch_policy = False

    @abstractmethod
    def get_cmdp(self) -> FiniteCMDP:
        raise NotImplementedError

    def setUp(self):
        self.cmdp = self.get_cmdp()
        if self.should_profile:
            """init each test"""
            self.pr = cProfile.Profile()
            self.pr.enable()

    def tearDown(self):
        if self.should_profile:
            """finish any test"""
            p = Stats(self.pr)
            p.strip_dirs()
            p.sort_stats('cumtime')
            p.print_stats()

    def test_solve(self):
        policy, solution_details = get_policy_solution_to_FiniteCMDP(self.cmdp)
        self.explore_solution(policy, solution_details)

    def explore_solution(self, policy, solution_details):
        explore_CMDP_solution_with_trajectories(policy, self.cmdp)


class TestSolveSimpleCAG(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        return MatrixCAGtoBCMDP(SimplestCAG())


class TestSolveRoseMazeCMDP(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        return RoseMazeCMDP()


class TestSolveRoseMazeCMDPStoch(TestCMDPSolver, TestCase):
    should_split_stoch_policy = True

    def get_cmdp(self) -> FiniteCMDP:
        return RoseMazeCMDP()


class TestSolveRoseGarden(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        return MatrixCAGtoBCMDP(RoseGarden())


class TestSolveCoordRoseGarden(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        return MatrixCAGtoBCMDP(CoopRoseGarden())


class TestTinySolveDCTFlora(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        cag = ForbiddenFloraDCTCoop(grid_size="tiny")
        return MatrixCAGtoBCMDP(cag)


class TestSmallSolveDCTFlora(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        cag = ForbiddenFloraDCTCoop(grid_size="small")
        return MatrixCAGtoBCMDP(cag)


class TestSolveDCTFlora(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        cag = ForbiddenFloraDCTCoop(grid_size="medium")
        return MatrixCAGtoBCMDP(cag)


class TestSolveDCTFloraLarge(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        cag = ForbiddenFloraDCTCoop(grid_size="large")
        return MatrixCAGtoBCMDP(cag)


class TestSolvePFDFlowers(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        cag = FlowerFieldPFDCoop()
        return MatrixCAGtoBCMDP(cag)


class TestSolvePFDFlowersSmall(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        cag = SmallFlowerFieldPFDCoop()
        return MatrixCAGtoBCMDP(cag)


class TestSolvePFDFlowersStochastic(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        cag = ForceStochasticFlowerFieldPFDCoop()
        return MatrixCAGtoBCMDP(cag)


class TestSolvePFDFlowersStochasticBreak(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        cag = BreaksReductionFlowerFieldPFDAppr()
        return MatrixCAGtoBCMDP(cag)


class TestDCTRoseGardenAppr(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        cag = DCTRoseGardenAppr()
        return MatrixCAGtoBCMDP(cag)


class TestSimplestFlowerFieldPFDCoop(TestCMDPSolver, TestCase):
    def get_cmdp(self) -> FiniteCMDP:
        cag = SimplestFlowerFieldPFDCoop()
        return MatrixCAGtoBCMDP(cag)
