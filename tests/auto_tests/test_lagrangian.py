import random
import unittest

import numpy as np

from src.example_environments.a_cmdp_that_requires_a_stochastic_policy import ACMDPNeedingStochasticity, \
    ASecondCMDPNeedingStochasticity
from src.example_environments.maze_cmdp import RoseMazeCMDP
from src.example_environments.randomised_cags_and_cmdps import RandomisedCMDP
from src.solvers.lagrangian_cmdp_solver import naive_lagrangian_cmdp_solver, find_minima_of_convex_f


class TestLagrangianSolver(unittest.TestCase):
    def test_convex_optim_is_close(self):
        offset = random.uniform(0.5, 3.5)

        def f(x):
            return (x - offset) ** 2

        estimated_offset = find_minima_of_convex_f(f, absolute_precision=1.e-9)
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
        _, _ = naive_lagrangian_cmdp_solver(cmdp, max_t=100)

    def test_lagrangian_runs_on_stoch1(self):
        """
        Note, this does not check for a close solution!
        :return:
        """
        cmdp = ACMDPNeedingStochasticity()
        _, _ = naive_lagrangian_cmdp_solver(cmdp, max_t=100)

    def test_lagrangian_runs_on_stoch2(self):
        """
        Note, this does not check for a close solution!
        :return:
        """
        cmdp = ASecondCMDPNeedingStochasticity()
        _, _ = naive_lagrangian_cmdp_solver(cmdp, max_t=100)

    def test_langrangian_runs_on_rand(self, K=1):
        """
        Note, this does not check for a close solution!
        :return:
        """
        randcmdp = RandomisedCMDP(K=K,
                                  max_steps=3,
                                  max_x=4,
                                  num_a=2)
        _, _ = naive_lagrangian_cmdp_solver(randcmdp, max_t=100)
