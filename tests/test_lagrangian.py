import random

from matplotlib import pyplot as plt

from src.formalisms import Lagrangian_CMDP_to_MDP
from src.particulars.maze_cmdp import RoseMazeCMDP
from src.particulars.rand_cag import RandomisedCMDP
from src.solvers.a_cmdp_that_requires_a_stochastic_policy import ACMDPNeedingStochasiticity, \
    ASecondCMDPNeedingStochasiticity
from src.solvers.lagrangian_cmdp_solver import naive_lagrangian_cmdp_solver, find_mimima_of_covex_f
from src.solvers.mdp_value_iteration import get_value_function_and_policy_by_iteration
from src.formalisms import CAG_to_BMDP, Distribution, CAG
from src.particulars.rose_garden_cag import RoseGarden


def test_convex_optim():
    offset = random.uniform(0.5, 3.5)

    def f(x):
        return (x - offset) ** 2

    find_mimima_of_covex_f(f)


def test_lagrangian_on_roses():
    cmdp = RoseMazeCMDP()
    mdp = Lagrangian_CMDP_to_MDP(cmdp, [1.0])
    vf = get_value_function_and_policy_by_iteration(mdp)
    naive_lagrangian_cmdp_solver(cmdp)


def test_lagrangian_on_stoch():
    cmdp = ACMDPNeedingStochasiticity()
    naive_lagrangian_cmdp_solver(cmdp)
    cmdp2 = ASecondCMDPNeedingStochasiticity()
    naive_lagrangian_cmdp_solver(cmdp2)

def test_langrangian_on_rand(K=1):
    randcmdp = RandomisedCMDP(K=K,
                              max_steps=3,
                              max_x=4,
                              num_a=2)
    naive_lagrangian_cmdp_solver(randcmdp)


if __name__ == "__main__":
    test_lagrangian_on_stoch()
    # test_lagrangian_on_roses()
    # test_langrangian_on_rand()
