import unittest

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

# class TestCAGtoCPOMDP(unittest.TestCase):
#
#     def test_cpomdp_to_mdp(self):
#         cag = RoseGarden()
#         cpomdp = CoordinationCPOMDP(cag)
#         cpomdp.perform_checks()
#         s0 = cpomdp.b_0.sample()
#         a0 = next(iter(cpomdp.A))
#         s1_dist = cpomdp.T(s0, a0)
#         s1 = s1_dist.sample()
#         reward = cpomdp.R(s0, a0)
