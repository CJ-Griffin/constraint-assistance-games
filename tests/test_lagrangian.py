import random
import unittest

from matplotlib import pyplot as plt

from src.formalisms.lagrangian_cmdp_to_mdp import Lagrangian_CMDP_to_MDP
from src.example_environments.maze_cmdp import RoseMazeCMDP
from src.example_environments.rand_cag import RandomisedCMDP
from src.solvers.a_cmdp_that_requires_a_stochastic_policy import ACMDPNeedingStochasiticity, \
    ASecondCMDPNeedingStochasiticity
from src.solvers.lagrangian_cmdp_solver import naive_lagrangian_cmdp_solver, find_mimima_of_covex_f
from src.solvers.mdp_value_iteration import get_value_function_and_policy_by_iteration


class TestLagrangianSolver(unittest.TestCase):
    def test_convex_optim(self):
        offset = random.uniform(0.5, 3.5)

        def f(x):
            return (x - offset) ** 2

        find_mimima_of_covex_f(f)

    def test_lagrangian_on_roses(self):
        cmdp = RoseMazeCMDP()
        mdp = Lagrangian_CMDP_to_MDP(cmdp, [1.0])
        vf = get_value_function_and_policy_by_iteration(mdp)
        naive_lagrangian_cmdp_solver(cmdp)

    def test_lagrangian_on_stoch(self):
        cmdp = ACMDPNeedingStochasiticity()
        naive_lagrangian_cmdp_solver(cmdp)
        cmdp2 = ASecondCMDPNeedingStochasiticity()
        naive_lagrangian_cmdp_solver(cmdp2)

    def test_langrangian_on_rand(self, K=1):
        randcmdp = RandomisedCMDP(K=K,
                                  max_steps=3,
                                  max_x=4,
                                  num_a=2)
        naive_lagrangian_cmdp_solver(randcmdp)
