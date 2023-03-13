from abc import ABC
from typing import List
from unittest import TestCase

import numpy as np

from src.concrete_decision_processes.ecas_examples.pfd_example import SimplestFlowerFieldPFDCoop
from src.concrete_decision_processes.maze_cmdp import RoseMazeCMDP
from src.concrete_decision_processes.randomised_cags_and_cmdps import RandomisedCAG, RandomisedCMDP
from src.formalisms.policy import FinitePolicyForFixedCMDP
from src.reductions.cag_to_bcmdp import CAGtoBCMDP
from src.solution_methods.linear_programming.cplex_dual_cmdp_solver import solve_CMDP_for_policy
from src.solution_methods.policy_splitting import split_policy


#
class TestPolicySplitter(TestCase):

    @staticmethod
    def get_cmdp():
        return RoseMazeCMDP()

    def setUp(self):
        self.cmdp = self.get_cmdp()
        self.cmdp.check_matrices()
        # TODO possibly add randomisation of never-states OR just determine their output
        self.sigma, _ = solve_CMDP_for_policy(self.cmdp)

    def test_splitter(self):
        policy, _ = solve_CMDP_for_policy(self.cmdp)
        phis, alphas = split_policy(self.sigma, self.cmdp)
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


class TestPolicySplitterCAG(TestPolicySplitter):
    @staticmethod
    def get_cmdp():
        cag = SimplestFlowerFieldPFDCoop()
        cmdp = CAGtoBCMDP(cag)
        return cmdp


class TestPolicySplitterRandomCMDP(TestPolicySplitter):
    @staticmethod
    def get_cmdp():
        cmdp = RandomisedCMDP()
        return cmdp


class TestPolicySplitterRandomCAG(TestPolicySplitter):
    @staticmethod
    def get_cmdp():
        cag = RandomisedCAG()
        cmdp = CAGtoBCMDP(cag)
        return cmdp
