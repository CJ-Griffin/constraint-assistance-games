import cProfile
import unittest
from abc import ABC, abstractmethod

from src.example_environments.ecas_examples.dct_example import ForbiddenFloraDCTApprenticeshipCAG
from src.example_environments.ecas_examples.pfd_example import FlowerFieldPrimaFacieDuties
from src.formalisms.cag import CAG, FiniteCAG
from src.formalisms.cag_to_bcmdp import MatrixCAGtoBCMDP
from src.formalisms.policy import RandomCAGPolicy
from src.solvers.linear_programming.cplex_dual_cmdp_solver import solve
from src.utils import explore_CAG_policy_with_env_wrapper


class TestECASCAG(ABC):
    profiled: bool

    def setUp(self):
        if self.profiled:
            self.pr = cProfile.Profile()
            self.pr.enable()

    def tearDown(self):
        if self.profiled:
            self.pr.disable()
            self.pr.print_stats(sort="tottime")

    @abstractmethod
    def create_process(self) -> FiniteCAG:
        pass

    def test_process(self):
        process = self.create_process()
        process.enable_debug_mode()

    def test_against_random(self):
        self.cag = self.create_process()
        self.policy = RandomCAGPolicy(S=self.cag.S, h_A=self.cag.h_A, r_A=self.cag.r_A)
        explore_CAG_policy_with_env_wrapper(self.policy, cag=self.cag)

    def test_by_solving(self):
        self.cag = self.create_process()
        print(self.cag.get_size_string())
        self.cmdp = MatrixCAGtoBCMDP(self.cag)
        print(self.cmdp.get_size_string())
        self.cmdp.check_matrices()
        print(self.cmdp.get_size_string())
        self.cmdp_policy, self.solution_details = solve(self.cmdp, )
        # explore_CMDP_solution_with_trajectories(self.cmdp_policy, self.cmdp)

    # def test_with_human(self):
    #     self.cag = self.create_process()
    #     explore_CAG_with_keyboard_input(self.cag)


class TestTinyDCTApprenticeshipCAG(TestECASCAG, unittest.TestCase):
    profiled = False

    def create_process(self) -> CAG:
        return ForbiddenFloraDCTApprenticeshipCAG(grid_size="tiny")


class TestTinyPFDApprenticeshipCAG(TestECASCAG, unittest.TestCase):
    profiled = False

    def create_process(self) -> CAG:
        return FlowerFieldPrimaFacieDuties(grid_size="tiny")
