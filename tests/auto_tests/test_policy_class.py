from unittest import TestCase

from src.concrete_decision_processes.randomised_cags_and_cmdps import RandomisedCMDP, NumlineState
from src.concrete_decision_processes.rose_garden_cags import SimplestCAG
from src.formalisms.distributions import UniformDiscreteDistribution, split_initial_dist_into_s_and_beta, \
    KroneckerDistribution
from src.formalisms.policy import FiniteCMDPPolicy, HistorySpace, RandomCAGPolicy, CAGPolicyFromCMDPPolicy, \
    RandomCMDPPolicy, FinitePolicyForFixedCMDP
from src.formalisms.primitives import FiniteSpace, IntState, IntAction
from src.utils.policy_analysis import explore_CAG_policy_with_env_wrapper
from src.reductions.cag_to_bcmdp import CAGtoBCMDP
from src.solution_methods.linear_programming.cplex_dual_cmdp_solver import solve_CMDP_for_policy


class TestPolicyClass(TestCase):
    def setUp(self):
        self.cmdp = RandomisedCMDP(max_x=2, max_steps=2)
        self.uni_dist = UniformDiscreteDistribution(self.cmdp.A)
        self.map = {s: self.uni_dist for s in self.cmdp.S}

    def test_normal(self):
        policy = FiniteCMDPPolicy(self.cmdp.S, self.cmdp.A, self.map)
        dist = policy(NumlineState(x=1, t=2))
        a = dist.sample()
        if a not in self.cmdp.A:
            raise ValueError

    def test_policy_rand_matrix(self):
        policy = FiniteCMDPPolicy(self.cmdp.S, self.cmdp.A, self.map)
        for s_ind, s in enumerate(self.cmdp.S):
            for a_ind, a in enumerate(self.cmdp.A):
                assert policy.policy_matrix[s_ind, a_ind] == policy(s).get_probability(a)

    def test_policy_det_matrix(self):
        default_action = list(self.cmdp.A)[0]
        action_dist = KroneckerDistribution(default_action)
        new_map = {s: action_dist for s in self.cmdp.S}
        policy = FiniteCMDPPolicy(self.cmdp.S, self.cmdp.A, new_map)
        for s_ind, s in enumerate(self.cmdp.S):
            for a_ind, a in enumerate(self.cmdp.A):
                assert policy.policy_matrix[s_ind, a_ind] == policy(s).get_probability(a)

    def test_occ_measure_generation(self):
        stat_dist_dict = {
            s: UniformDiscreteDistribution(self.cmdp.A)
            for s in self.cmdp.S
        }
        policy = FinitePolicyForFixedCMDP(self.cmdp, stat_dist_dict)
        policy._generate_occupancy_measure_matrix(should_validate=True)
        

class TestCAGPolicyGenerator(TestCase):

    def setUp(self):
        self.cag = SimplestCAG()
        self.cmdp = CAGtoBCMDP(self.cag)
        self.cmdp.check_matrices()
        self.cmdp_policy, self.solution_details = solve_CMDP_for_policy(self.cmdp)

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
