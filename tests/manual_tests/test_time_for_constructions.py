import unittest

from src.concrete_processes.rose_garden_cags import RoseGarden
from src.utils.utils import time_function


# from abc import ABC, abstractmethod
# from src.formalisms.cag import CAG
# from src.formalisms.cag_to_bcmdp import CAGtoBCMDP, split_initial_dist_into_s_and_beta
# from src.formalisms.policy import CAGPolicyFromCMDPPolicy
# from src.solution_methods.linear_programming.cplex_dual_cmdp_solver import solve
# from src.utils import explore_CAG_policy_with_env_wrapper


class TestRewardMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.cag = RoseGarden()

    def test_reward_matrix(self):
        self.cag.generate_matrices()
        self.cag.check_matrices()

        s, h_a, r_a = next(iter(self.cag.S)), next(iter(self.cag.h_A)), next(iter(self.cag.r_A))

        print()

        @time_function
        def time_standard_reward_get():
            for i in range(1000):
                _ = self.cag._inner_split_R(s, h_a, r_a)

        time_standard_reward_get()

        s_ind, h_a_in, r_a_ind = self.cag.state_to_ind_map[s], self.cag.human_action_to_ind_map[h_a], \
                                 self.cag.robot_action_to_ind_map[r_a]

        @time_function
        def time_matrix_reward_get_1():
            for i in range(1000):
                _ = self.cag.reward_matrix_s_ha_ra[s_ind, h_a_in, r_a_ind]

        time_matrix_reward_get_1()

        @time_function
        def time_matrix_reward_get_2():
            for i in range(1000):
                _ = self.cag.reward_matrix_s_ha_ra[self.cag.state_to_ind_map[s], self.cag.human_action_to_ind_map[h_a], \
                                                   self.cag.robot_action_to_ind_map[r_a]]

        time_matrix_reward_get_2()

