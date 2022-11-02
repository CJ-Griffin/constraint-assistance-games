from unittest import TestCase

from src.concrete_processes.maze_cmdp import RoseMazeCMDP
from src.concrete_processes.rose_garden_cag import RoseGarden
from src.concrete_processes.simplest_cag import SimplestCAG
from src.reductions.cag_to_bcmdp import CAGtoBCMDP
from src.formalisms.policy import HistorySpace, CMDPPolicy, RandomCAGPolicy, CAGPolicyFromCMDPPolicy
from src.get_traj_dist import get_traj_dist
from src.solvers.linear_programming.cplex_dual_cmdp_solver import solve
from src.policy_analysis import explore_CMDP_solution_with_trajectories, explore_CMDP_solution_extionsionally, \
    explore_CMDP_policy_with_env_wrapper, explore_CAG_policy_with_env_wrapper
from src.formalisms.distributions import split_initial_dist_into_s_and_beta


class TestCMDPSolver(TestCase):
    def setUp(self):
        self.cmdp = RoseMazeCMDP()
        self.cmdp.check_matrices()

    def test_solve(self):
        """
        Note these tests are for funding Exceptions and *not* for testing validity of solutions.
        :return:
        """
        policy, _ = solve(self.cmdp)
        explore_CMDP_solution_with_trajectories(policy, self.cmdp)


class TestDistGenerator(TestCMDPSolver):
    def test_traj_dist_generator(self):
        self.policy, self.solution_details = solve(self.cmdp)
        get_traj_dist(cmdp=self.cmdp, pol=self.policy)


class TestWrapperBasedExplorer(TestCMDPSolver):
    def test_wrapper_based_explorer(self):
        pair = solve(self.cmdp)
        self.policy: CMDPPolicy = pair[0]
        self.solution_details = pair[1]
        explore_CMDP_policy_with_env_wrapper(policy=self.policy, cmdp=self.cmdp)


class TestExtensionalExplorer(TestCMDPSolver):
    def test_wrapper_based_explorer(self):
        pair = solve(self.cmdp)
        self.policy: CMDPPolicy = pair[0]
        self.solution_details = pair[1]
        explore_CMDP_solution_extionsionally(policy=self.policy,
                                             solution_details=self.solution_details,
                                             supress_print=True)


class TestDualSolveSimpleCAG(TestCMDPSolver):
    def setUp(self):
        self.cag = RoseGarden()
        self.cmdp = CAGtoBCMDP(self.cag)
        self.cmdp.check_matrices()


class TestCAGPolicyGenerator(TestCase):

    def setUp(self):
        self.cag = SimplestCAG()
        self.cmdp = CAGtoBCMDP(self.cag)
        self.cmdp.check_matrices()
        self.cmdp_policy, self.solution_details = solve(self.cmdp)

    def test_history_space(self):
        hist_space = HistorySpace(S=self.cag.S, A=self.cag.A)
        states = list(self.cag.S)
        i_range = range(len(states) * 20)
        hist_sample = []
        for i, hist in zip(i_range, hist_space):
            hist.validate_for_process(self.cag)
            hist_sample.append(hist)
        zero_length_samples = [h for h in hist_sample if h.t == 0]

        if len(states) != len(zero_length_samples):
            raise ValueError

    def test_cag_policy_class(self):
        random_policy = RandomCAGPolicy(S=self.cag.S, h_A=self.cag.h_A, r_A=self.cag.r_A)
        explore_CAG_policy_with_env_wrapper(random_policy, self.cag, should_render=True)

    def test_cmdp_to_cag_policy_converter(self):
        _, beta_0 = split_initial_dist_into_s_and_beta(self.cag.initial_state_theta_dist)
        cag_policy = CAGPolicyFromCMDPPolicy(cmdp_policy=self.cmdp_policy, beta_0=beta_0)
        explore_CAG_policy_with_env_wrapper(cag_policy, self.cag, should_render=True)


class TestCAGPolicyGeneratorOnBasicStochastic(TestCase):
    def setUp(self):
        self.cag = SimplestCAG(budget=0.1)
        self.cmdp = CAGtoBCMDP(self.cag)
        self.cmdp.check_matrices()
        self.cmdp_policy, self.solution_details = solve(self.cmdp)

    def test_cmdp_to_cag_policy_converter(self):
        _, beta_0 = split_initial_dist_into_s_and_beta(self.cag.initial_state_theta_dist)
        try:
            cag_policy = CAGPolicyFromCMDPPolicy(cmdp_policy=self.cmdp_policy, beta_0=beta_0)
            explore_CAG_policy_with_env_wrapper(cag_policy, self.cag, should_render=True)
        except NotImplementedError as e:
            print(e)
