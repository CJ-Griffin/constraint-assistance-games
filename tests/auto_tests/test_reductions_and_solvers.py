import random
import unittest
from unittest import TestCase

import numpy as np

from src.concrete_decision_processes.a_cmdp_that_requires_a_stochastic_policy import ACMDPNeedingStochasticity, \
    ASecondCMDPNeedingStochasticity
from src.concrete_decision_processes.maze_cmdp import RoseMazeCMDP
from src.concrete_decision_processes.randomised_cags_and_cmdps import RandomisedCMDP
from src.concrete_decision_processes.rose_garden_cags import SimplestCAG
from src.formalisms.finite_processes import FiniteCAG
from src.reductions.cag_to_bcmdp import CAGtoBCMDP, MatrixCAGtoBCMDP
from src.reductions.lagrangian_cmdp_to_mdp import LagrangianCMDPtoMDP
from src.solution_methods.lagrangian_cmdp_solver import find_minima_of_convex_f, \
    get_value_function_using_naive_lagrangian_cmdp_solver
from src.solution_methods.solvers import get_policy_solution_to_FiniteCMDP
from src.utils.get_traj_dist import get_traj_dist
from src.utils.policy_analysis import explore_CMDP_solution_with_trajectories, explore_CMDP_solution_extionsionally, \
    explore_CMDP_policy_with_env_wrapper
from src.utils.utils import raise_exception_at_difference_in_arrays


class TestCMDPSolver(TestCase):
    def setUp(self):
        self.cmdp = RoseMazeCMDP()
        self.cmdp.check_matrices()

    def test_solve(self):
        """
        Note these tests are for finding Exceptions and *not* for testing validity of solutions.
        :return:
        """
        policy = get_policy_solution_to_FiniteCMDP(self.cmdp)
        explore_CMDP_solution_with_trajectories(policy, self.cmdp)


class TestCMDPSolutionExplorers(TestCase):
    def setUp(self):
        self.cmdp = RoseMazeCMDP()
        self.cmdp.check_matrices()
        self.policy, self.solution_details = get_policy_solution_to_FiniteCMDP(self.cmdp)

    def test_wrapper_based_explorer(self):
        explore_CMDP_policy_with_env_wrapper(policy=self.policy, cmdp=self.cmdp)

    def test_traj_dist_generator(self):
        get_traj_dist(cmdp=self.cmdp, pol=self.policy)

    def test_traj_based_explorer(self):
        explore_CMDP_solution_extionsionally(policy=self.policy,
                                             solution_details=self.solution_details,
                                             supress_print=True)


class TestLagrangianReduction(TestCase):
    def test_mdp_runs(self):
        cmdp = RoseMazeCMDP()
        mdp = LagrangianCMDPtoMDP(cmdp, lagrange_multiplier=[10.0])
        mdp.perform_checks()


class TestCAGtoCMDP(TestCase):

    def test_cag_to_bcmdp(self):
        cag = SimplestCAG()
        cmdp = CAGtoBCMDP(cag)
        cmdp.check_matrices()
        cmdp.perform_checks()

    def test_cag_to_bcmdp_via_matrices(self):
        cag1: FiniteCAG = SimplestCAG()
        cag2: FiniteCAG = SimplestCAG()
        numpy_bcmdp = MatrixCAGtoBCMDP(cag1)
        old_bcmdp = CAGtoBCMDP(cag2)

        def test_numpy():
            numpy_bcmdp.initialise_matrices()

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


class TestLagrangianSolver(unittest.TestCase):
    def test_convex_optim_is_close(self):
        offset = random.uniform(0.5, 3.5)

        def f(x):
            return (x - offset) ** 2

        estimated_offset, _ = find_minima_of_convex_f(f, absolute_precision=1.e-9)
        if np.isclose(estimated_offset, offset):
            pass
        else:
            raise ValueError

    def test_lagrangian_runs_on_roses(self):
        """
        Note, this does not check for a close solution!
        :return:
        """
        cmdp = RoseMazeCMDP()
        _, _ = get_value_function_using_naive_lagrangian_cmdp_solver(cmdp, max_t=100)

    def test_lagrangian_runs_on_stoch1(self):
        """
        Note, this does not check for a close solution!
        :return:
        """
        cmdp = ACMDPNeedingStochasticity()
        _, _ = get_value_function_using_naive_lagrangian_cmdp_solver(cmdp, max_t=100)

    def test_lagrangian_runs_on_stoch2(self):
        """
        Note, this does not check for a close solution!
        :return:
        """
        cmdp = ASecondCMDPNeedingStochasticity()
        _, _ = get_value_function_using_naive_lagrangian_cmdp_solver(cmdp, max_t=100)

    def test_langrangian_runs_on_rand(self, K=1):
        """
        Note, this does not check for a close solution!
        :return:
        """
        randcmdp = RandomisedCMDP(K=K,
                                  max_steps=3,
                                  max_x=4,
                                  num_a=2)
        _, _ = get_value_function_using_naive_lagrangian_cmdp_solver(randcmdp, max_t=100)
