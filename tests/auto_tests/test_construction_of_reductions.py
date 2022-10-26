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

        m1 = cmdp1.transition_matrix
        m2 = cmdp2.transition_matrix

        if not (m1 == m2).all():
            self.raise_exception_at_difference(cmdp1, m1, m2)

    @staticmethod
    def raise_exception_at_difference(cmdp1, m1, m2):
        import numpy as np
        locs = np.argwhere(m1 != m2)
        for i in range(locs.shape[0]):
            b_ind, a_ind, b_next_ind = locs[i, :]
            b = cmdp1.state_list[b_ind]
            a = cmdp1.action_list[a_ind]
            b_next = cmdp1.state_list[b_next_ind]
            p1 = m1[b_ind, a_ind, b_next_ind]
            p2 = m2[b_ind, a_ind, b_next_ind]

            x1 = b[0]
            x2 = b[1]

            x3 = a[0]
            x4 = a[1]

            x5 = b_next[0]
            x6 = b_next[1]

            x7 = p1

            x8 = p2

            raise ValueError
