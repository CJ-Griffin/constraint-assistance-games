from unittest import TestCase

import numpy as np

from src.concrete_decision_processes.randomised_cags_and_cmdps import RandomisedCMDP, NumlineState
from src.concrete_decision_processes.rose_garden_cags import SimplestCAG
from src.formalisms.distributions import UniformDiscreteDistribution, split_initial_dist_into_s_and_beta, \
    KroneckerDistribution
from src.formalisms.policy import DictCMDPPolicy, HistorySpace, RandomCAGPolicy, CAGPolicyFromCMDPPolicy, \
    RandomCMDPPolicy, FinitePolicyForFixedCMDP
from src.formalisms.primitives import FiniteSpace, IntState, IntAction
from src.solution_methods.solvers import get_policy_solution_to_FiniteCMDP
from src.utils.policy_analysis import explore_CAG_policy_with_env_wrapper
from src.reductions.cag_to_bcmdp import CAGtoBCMDP


class TestPolicyClass(TestCase):
    def setUp(self):
        self.cmdp = RandomisedCMDP(max_x=2, max_steps=2)
        self.uni_dist = UniformDiscreteDistribution(self.cmdp.A)
        self.map = {s: self.uni_dist for s in self.cmdp.S}

    def test_normal(self):
        policy = DictCMDPPolicy(self.cmdp.S, self.cmdp.A, self.map)
        dist = policy(NumlineState(x=1, t=2))
        a = dist.sample()
        if a not in self.cmdp.A:
            raise ValueError

    def test_occ_measure_generation(self):
        stat_dist_dict = {
            s: UniformDiscreteDistribution(self.cmdp.A)
            for s in self.cmdp.S
        }
        policy1 = FinitePolicyForFixedCMDP.fromPolicyDict(self.cmdp, stat_dist_dict, True)
        policy2 = FinitePolicyForFixedCMDP.fromPolicyMatrix(self.cmdp, policy1.policy_matrix)
        occupancy_measures = policy2.occupancy_measure_matrix
        policy3 = FinitePolicyForFixedCMDP.fromOccupancyMeasureMatrix(self.cmdp, occupancy_measures)

        # They can be dissimilar, so long as the state has no chance of being entered
        for s_ind in range(self.cmdp.n_states):
            probs1 = policy1.policy_matrix[s_ind, :]
            probs3 = policy3.policy_matrix[s_ind, :]
            if not np.allclose(probs1, probs3):
                if occupancy_measures[s_ind, :].sum() != 0.0:
                    raise ValueError


class TestCAGPolicyGenerator(TestCase):

    def setUp(self):
        self.cag = SimplestCAG()
        self.cmdp = CAGtoBCMDP(self.cag)
        self.cmdp.check_matrices()
        self.cmdp_policy = get_policy_solution_to_FiniteCMDP(self.cmdp)

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
