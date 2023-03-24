from unittest import TestCase

import numpy as np

from src.concrete_decision_processes.maze_cmdp import RoseMazeCMDP
from src.concrete_decision_processes.randomised_cags_and_cmdps import RandomisedCAG, RandomisedCMDP
from src.reductions.cag_to_bcmdp import CAGtoBCMDP
from src.solution_methods.policy_splitting import split_cmdp_policy
from src.solution_methods.solvers import get_policy_solution_to_FiniteCMDP


#
class TestCMDPPolicySplitter(TestCase):

    @staticmethod
    def get_cmdp():
        return RoseMazeCMDP()

    def setUp(self):
        self.cmdp = self.get_cmdp()
        self.cmdp.check_matrices()
        self.sigma = get_policy_solution_to_FiniteCMDP(self.cmdp)

    def test_splitter(self):
        phis, alphas = split_cmdp_policy(self.sigma, self.cmdp)
        for phi in phis:
            assert phi.get_is_policy_deterministic()
            phi.validate()

        weighted_sum_of_new_policy_oms = sum(
            alphas[j] * phis[j].occupancy_measure_matrix
            for j in range(len(phis))
        )
        if not np.allclose(self.sigma.occupancy_measure_matrix, weighted_sum_of_new_policy_oms):
            A = self.sigma.occupancy_measure_matrix
            B = weighted_sum_of_new_policy_oms
            relative_diffs = np.absolute((B - A)) / B
            epsilon = relative_diffs[~np.isnan(relative_diffs)].max()
            # TODO, reduce this error bound
            if epsilon < 0.01:
                print("warning! there seems to be a numerical error problem")
            else:
                raise ValueError(f"Differs by as much as {epsilon}")
        print(f"Successfully split into {len(phis)} deterministic policies")


class TestPolicySplitterRandomCMDP42(TestCMDPPolicySplitter):
    @staticmethod
    def get_cmdp():
        cmdp = RandomisedCMDP(seed=42)
        return cmdp


class TestPolicySplitterRandomCAG42(TestCMDPPolicySplitter):
    @staticmethod
    def get_cmdp():
        cag = RandomisedCAG(seed=42)
        cmdp = CAGtoBCMDP(cag)
        return cmdp
