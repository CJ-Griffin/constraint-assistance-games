import unittest

from src.example_environments.ecas_examples import ForbiddenFloraDCTApprenticeshipCAG
from src.example_environments.maze_cmdp import RoseMazeCMDP
from src.example_environments.simplest_cag import SimplestCAG
from src.formalisms.cag_to_bcmdp import CAGtoBCMDP, MatrixCAGtoBCMDP
# from src.formalisms.cag_to_cpomdp import CoordinationCPOMDP
from src.formalisms.lagrangian_cmdp_to_mdp import Lagrangian_CMDP_to_MDP


class TestLagrangianReduction(unittest.TestCase):
    def test_mdp_runs(self):
        cmdp = RoseMazeCMDP()
        mdp = Lagrangian_CMDP_to_MDP(cmdp, lagrange_multiplier=[10.0])
        mdp.perform_checks()
        s_0 = mdp.initial_state_dist.sample()
        a_0 = next(iter(mdp.A))
        s_1_dist = mdp.T(s_0, a_0)
        s_1 = s_1_dist.sample()
        reward = mdp.R(s_0, a_0)


class TestCAGtoCMDP(unittest.TestCase):

    def test_cag_to_bcmdp(self):
        cag = SimplestCAG()
        cmdp = CAGtoBCMDP(cag)
        cmdp.check_matrices()
        cmdp.perform_checks()
        s_0 = cmdp.initial_state_dist.sample()
        a_0 = next(iter(cmdp.A))
        s_1_dist = cmdp.T(s_0, a_0)
        s_1 = s_1_dist.sample()
        reward = cmdp.R(s_0, a_0)


class TestCAGtoCMDPViaMatrices(unittest.TestCase):

    def test_cag_to_bcmdp_via_matrices(self):
        cag1 = SimplestCAG()
        cag2 = SimplestCAG()
        cmdp1 = MatrixCAGtoBCMDP(cag1)
        cmdp2 = MatrixCAGtoBCMDP(cag2)

        cmdp1._initialise_matrices_new()
        cmdp2._initialise_matrices_old()

        if cmdp1.beta_list != cmdp2.beta_list:
            raise ValueError
        elif cmdp1.state_list != cmdp2.state_list:
            raise ValueError
        elif cmdp1.action_list != cmdp2.action_list:
            raise ValueError

        self.raise_exception_at_difference(cmdp1, cmdp1.transition_matrix, cmdp2.transition_matrix)
        self.raise_exception_at_difference(cmdp1, cmdp1.reward_matrix, cmdp2.reward_matrix)
        self.raise_exception_at_difference(cmdp1, cmdp1.cost_matrix, cmdp2.cost_matrix)

    def time_cag_to_bcmdp_via_matrices(self):
        cag1 = ForbiddenFloraDCTApprenticeshipCAG()
        cag2 = ForbiddenFloraDCTApprenticeshipCAG()
        cmdp1 = MatrixCAGtoBCMDP(cag1)
        cmdp2 = MatrixCAGtoBCMDP(cag2)

        cmdp1._initialise_matrices_new()
        cmdp2._initialise_matrices_old()

    @staticmethod
    def raise_exception_at_difference(cmdp1, m1, m2):
        if not m1.shape == m2.shape:
            s1 = m1.shape
            s2 = m2.shape
            raise Exception
        if not (m1 == m2).all():
            import numpy as np
            locs = np.argwhere(m1 != m2)
            for i in range(locs.shape[0]):
                triplet = locs[i]
                v1 = m1[tuple(triplet)]
                v2 = m2[tuple(triplet)]
                raise ValueError
