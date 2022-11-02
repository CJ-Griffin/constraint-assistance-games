import unittest

from src.concrete_processes.maze_cmdp import RoseMazeCMDP
from src.concrete_processes.simplest_cag import SimplestCAG
from src.formalisms.finite_processes import FiniteCAG
from src.reductions.cag_to_bcmdp import CAGtoBCMDP, MatrixCAGtoBCMDP
# from src.formalisms.cag_to_cpomdp import CoordinationCPOMDP
from src.reductions.lagrangian_cmdp_to_mdp import LagrangianCMDPtoMDP
from src.utils import raise_exception_at_difference_in_arrays, time_function


class TestLagrangianReduction(unittest.TestCase):
    def test_mdp_runs(self):
        cmdp = RoseMazeCMDP()
        mdp = LagrangianCMDPtoMDP(cmdp, lagrange_multiplier=[10.0])
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

    @staticmethod
    def get_cag():
        return SimplestCAG()

    def test_cag_to_bcmdp_via_matrices(self):
        cag1: FiniteCAG = self.get_cag()
        cag2: FiniteCAG = self.get_cag()
        numpy_bcmdp = MatrixCAGtoBCMDP(cag1)
        old_bcmdp = CAGtoBCMDP(cag2)

        @time_function
        def test_numpy():
            numpy_bcmdp.initialise_matrices()

        @time_function
        def test_old():
            old_bcmdp.initialise_matrices()

        test_numpy()
        test_old()

        if numpy_bcmdp.state_list != old_bcmdp.state_list:
            raise ValueError
        elif numpy_bcmdp.action_list != old_bcmdp.action_list:
            raise ValueError

        raise_exception_at_difference_in_arrays(numpy_bcmdp.transition_matrix, old_bcmdp.transition_matrix)
        raise_exception_at_difference_in_arrays(numpy_bcmdp.reward_matrix, old_bcmdp.reward_matrix)
        raise_exception_at_difference_in_arrays(numpy_bcmdp.cost_matrix, old_bcmdp.cost_matrix)
