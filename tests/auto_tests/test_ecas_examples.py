import unittest
from abc import ABC, abstractmethod

from src.example_environments.ecas_examples import ForbiddenFloraDCTApprenticeshipCAG
from src.formalisms.cag import CAG
from src.formalisms.cag_to_bcmdp import CAGtoBCMDP, MatrixCAGtoBCMDP
from src.solvers.linear_programming.cplex_dual_cmdp_solver import solve


class TestECASCAG(ABC):
    @abstractmethod
    def create_process(self) -> CAG:
        pass

    # def test_against_random(self):
    #     self.cag = self.create_process()
    #     self.policy = RandomCAGPolicy(S=self.cag.S, h_A=self.cag.h_A, r_A=self.cag.r_A)
    #     explore_CAG_policy_with_env_wrapper(self.policy, cag=self.cag)

    def test_by_solving(self):
        self.cag = self.create_process()
        print(self.cag.get_size_string())
        self.cmdp = MatrixCAGtoBCMDP(self.cag)
        print(self.cmdp.get_size_string())
        self.cmdp.check_matrices()
        print(self.cmdp.get_size_string())
        self.cmdp_policy, self.solution_details = solve(self.cmdp, )
        # _, beta_0 = split_initial_dist_into_s_and_beta(self.cag.initial_state_theta_dist)
        # cag_policy = CAGPolicyFromCMDPPolicy(cmdp_policy=self.cmdp_policy, beta_0=beta_0)
        # explore_CAG_policy_with_env_wrapper(cag_policy, self.cag, should_render=True)


class TestTinyDCTApprenticeshipCAG(TestECASCAG, unittest.TestCase):
    def create_process(self) -> CAG:
        return ForbiddenFloraDCTApprenticeshipCAG(grid_size="tiny")
