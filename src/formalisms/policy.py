import itertools
from abc import abstractmethod
from typing import List, FrozenSet, Tuple

import numpy as np

from src.formalisms.distributions import Distribution, UniformDiscreteDistribution, FiniteParameterDistribution, \
    KroneckerDistribution, DiscreteDistribution
from src.formalisms.finite_processes import FiniteCMDP
from src.formalisms.primitives import State, ActionPair, Action, Plan, Space, FiniteSpace, CountableSpace
from src.formalisms.trajectory import Trajectory
from src.reductions.cag_to_bcmdp import BeliefState


class CMDPPolicy:
    def __init__(self, S: Space, A: FrozenSet[Action]):
        self.S: Space = S
        self.A: FrozenSet[Action] = A

    @abstractmethod
    def _get_distribution(self, s: State) -> Distribution:
        pass

    def __call__(self, s: State) -> Distribution:
        if s not in self.S:
            raise ValueError
        else:
            return self._get_distribution(s)


class FiniteCMDPPolicy(CMDPPolicy):

    def __init__(self, S: Space, A: FrozenSet[Action], state_to_dist_map: dict):
        if not isinstance(S, FiniteSpace):
            raise ValueError
        super().__init__(S, A)
        self._state_to_dist_map = state_to_dist_map
        self.n_states = len(S)
        self.n_actions = len(A)
        self._policy_matrix: np.array = None
        self.validate()

    def _get_distribution(self, s: State) -> Distribution:
        return self._state_to_dist_map[s]

    def validate(self):
        for s in self.S:
            if s not in self._state_to_dist_map:
                raise ValueError
            else:
                for a in self._state_to_dist_map[s].support():
                    if a not in self.A:
                        raise ValueError(a, self.A)

    def _generate_policy_matrix(self):
        self._policy_matrix = np.zeros((self.n_states, self.n_actions))
        for s_ind, s in enumerate(self.S):
            for a_ind, a in enumerate(self.A):
                next_state_dist = self(s)
                if next_state_dist is None:
                    raise ValueError(s, self._state_to_dist_map)
                prob = next_state_dist.get_probability(a)
                self._policy_matrix[s_ind, a_ind] = prob
        assert np.allclose(self._policy_matrix.sum(axis=1), 1.0)

    @property
    def policy_matrix(self):
        if self._policy_matrix is None:
            self._generate_policy_matrix()
        return self._policy_matrix


class FinitePolicyForFixedCMDP(FiniteCMDPPolicy):
    def __init__(
            self,
            cmdp: FiniteCMDP,
            policy_matrix: np.ndarray,
            occupancy_measure_matrix: np.ndarray
    ):
        policy_dict = dict()
        for s in cmdp.S:
            action_prob_dict = {
                a: policy_matrix[cmdp.state_to_ind_map[s], cmdp.action_to_ind_map[a]]
                for a in cmdp.A
            }
            policy_dict[s] = DiscreteDistribution(action_prob_dict)
        super().__init__(S=cmdp.S, A=cmdp.A, state_to_dist_map=policy_dict)
        self._policy_matrix = policy_matrix
        self.occupancy_measure_matrix = occupancy_measure_matrix
        self.cmdp = cmdp

    @classmethod
    def fromPolicyDict(
            cls,
            cmdp: FiniteCMDP,
            policy_dict: dict,
            should_cross_reference_methods: bool = False
    ):
        cmdp.initialise_matrices()
        policy_matrix = np.zeros((cmdp.n_states, cmdp.n_actions))
        for s in cmdp.S:
            s_ind = cmdp.state_to_ind_map[s]
            a_dist = policy_dict[s]
            for a in a_dist.support():
                a_ind = cmdp.action_to_ind_map[a]
                policy_matrix[s_ind, a_ind] = a_dist.get_probability(a)
        return FinitePolicyForFixedCMDP.fromPolicyMatrix(cmdp, policy_matrix, should_cross_reference_methods)

    @classmethod
    def fromPolicyMatrix(cls,
                         cmdp: FiniteCMDP,
                         policy_matrix: np.ndarray,
                         should_cross_reference_methods: bool = False):
        assert policy_matrix.shape == (cmdp.n_states, cmdp.n_actions)
        assert np.allclose(policy_matrix.sum(axis=1), 1)
        assert (policy_matrix >= 0).all()

        occupancy_measure_matrix = FinitePolicyForFixedCMDP._calculate_occupancy_measure_matrix(
            cmdp=cmdp,
            policy_matrix=policy_matrix,
            should_cross_reference_methods=should_cross_reference_methods
        )
        return FinitePolicyForFixedCMDP(cmdp, policy_matrix, occupancy_measure_matrix)

    @staticmethod
    def _calculate_occupancy_measure_matrix(
            cmdp: FiniteCMDP,
            policy_matrix: np.ndarray,
            should_cross_reference_methods: bool = False):

        state_vector = FinitePolicyForFixedCMDP._analytically_calculate_state_occupancy_measures(
            cmdp=cmdp,
            policy_matrix=policy_matrix
        )

        if should_cross_reference_methods:
            approximate_state_vector = FinitePolicyForFixedCMDP._iteratively_approimate_state_occupancy_measures(
                cmdp=cmdp,
                policy_matrix=policy_matrix
            )
            if not np.allclose(approximate_state_vector, state_vector):
                raise ValueError(f"Max difference of {(approximate_state_vector - state_vector).max()}.",
                                 "Maybe increase the number of iterations?")

        # Check that q is a reasonable value
        # ∑_i q[i] = ∑_i ∑_t γ^t P[S_t = s_i]
        #          = ∑_t γ^t * 1
        #          = 1 - (1-γ)
        assert np.isclose(state_vector.sum(), (1.0 / (1 - cmdp.gamma)))

        occupancy_measure_matrix = policy_matrix * state_vector.reshape(-1, 1)
        assert occupancy_measure_matrix.shape == (cmdp.n_states, cmdp.n_actions)
        return occupancy_measure_matrix

    @staticmethod
    def _iteratively_approimate_state_occupancy_measures(
            cmdp: FiniteCMDP,
            policy_matrix: np.ndarray,
            num_iterations: int = 10000
    ) -> np.ndarray:
        """
        Use the series q_{t+1} = d_0 + γ (q_t @ P) described in src/notebooks/calculating_occupancies.ipynb
        :param self:
        :param num_iterations: the number of times to iterate over the series, it takes longer to iterate more times,
        but the result will be more accurate
        :return: An approximation of q ∈ R^{n_states}, q[i] = ∑_t γ^t P[S_t = s_i]
        """
        d_0 = cmdp.start_state_probabilities
        q_t = d_0

        # P[i,j] = Σ T[i, a, j] π[i, a] = P[S_t+1 = s_j | S_t = s_i, π]
        P = np.einsum('iaj,ia->ij', cmdp.transition_matrix, policy_matrix)
        assert P.shape == (cmdp.n_states, cmdp.n_states)
        assert np.allclose(P.sum(axis=1), 1.0)

        assert num_iterations >= 1
        # sums = [d_0.sum()]
        for t in range(num_iterations):
            q_t = d_0 + (cmdp.gamma * (q_t @ P))
            # sums.append(q_t.sum())
        return q_t

    @staticmethod
    def _analytically_calculate_state_occupancy_measures(
            cmdp: FiniteCMDP,
            policy_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Exact calculation using matrix inv described in src/notebooks/calculating_occupancies.ipynb
        :param self:
        :return: q ∈ R^{n_states}, q[i] = ∑_t γ^t P[S_t = s_i]
        """
        d_0 = cmdp.start_state_probabilities
        q_t = d_0

        # P[i,j] = Σ T[i, a, j] π[i, a] = P[S_t+1 = s_j | S_t = s_i, π]
        P = np.einsum('iaj,ia->ij', cmdp.transition_matrix, policy_matrix)
        assert P.shape == (cmdp.n_states, cmdp.n_states)
        assert np.allclose(P.sum(axis=1), 1.0)

        try:
            inverse = np.linalg.inv((np.identity(cmdp.n_states) - (cmdp.gamma * P)))
            q = d_0 @ inverse
            return q
        except np.linalg.LinAlgError as lin_alg_error:
            NotImplementedError("Matrix inversion failed! You need to define what to do in this case.", lin_alg_error)


class RandomCMDPPolicy(CMDPPolicy):

    def __init__(self, S: Space, A: FrozenSet[Action]):
        if not isinstance(S, FiniteSpace):
            raise ValueError
        super().__init__(S, A)

    def _get_distribution(self, s: State) -> Distribution:
        return UniformDiscreteDistribution(self.A)


class HistorySpaceIterator:
    def __init__(self, S: Space, A: FrozenSet[Action]):
        self.cur_t = 0
        self.S = tuple(S)
        self.A = tuple(A)
        self.cur_state_action_tuple_iterator: iter = self.get_t_th_iter_for_state_action_tuples(t=0)

    def __next__(self):
        try:
            next_tuple_pair = next(self.cur_state_action_tuple_iterator)
        except StopIteration as e:
            self.cur_t += 1
            self.cur_state_action_tuple_iterator = self.get_t_th_iter_for_state_action_tuples(t=self.cur_t)
            next_tuple_pair = next(self.cur_state_action_tuple_iterator)
        states, actions = next_tuple_pair
        return Trajectory(t=self.cur_t, states=states, actions=actions)

    def get_t_th_iter_for_state_action_tuples(self, t: int):
        state_iterators = [
            self.get_state_iterator() for i in range(t + 1)
        ]
        action_iterators = [self.get_action_iterator() for i in range(t)]

        state_tuple_iterator = itertools.product(*state_iterators)
        action_tuple_iterator = itertools.product(*action_iterators)
        return itertools.product(state_tuple_iterator, action_tuple_iterator)

    def get_state_iterator(self) -> iter:
        return iter(self.S)

    def get_action_iterator(self) -> iter:
        return iter(self.A)


class HistorySpace(CountableSpace):
    def __init__(self, S: FiniteSpace, A: FrozenSet[Action]):
        self.S = S
        self.A = A

    def __contains__(self, hist: Trajectory):
        if not isinstance(hist, Trajectory):
            return False
        elif not hist.get_whether_actions_in_A(self.A):
            return False
        elif not hist.get_whether_states_in_S(self.S):
            return False
        else:
            return True

    def __iter__(self) -> iter:
        return HistorySpaceIterator(S=self.S, A=self.A)


class FiniteCAGPolicy:
    def __init__(self, S: FiniteSpace, h_A: FrozenSet[Action], r_A: FrozenSet[Action]):
        self.S: FiniteSpace = S
        self.A_joint_concrete: frozenset = frozenset({ActionPair(h_a, r_a) for h_a in h_A for r_a in r_A})
        self.hist_space = HistorySpace(S, self.A_joint_concrete)

    @abstractmethod
    def _get_distribution(self, hist: Trajectory, theta) -> Distribution:
        pass

    def __call__(self, hist: Trajectory, theta) -> Distribution:
        if hist not in self.hist_space:
            raise ValueError
        else:
            return self._get_distribution(hist, theta)


class RandomCAGPolicy(FiniteCAGPolicy):

    def _get_distribution(self, hist: Trajectory, theta) -> Distribution:
        return UniformDiscreteDistribution(self.A_joint_concrete)


class CAGPolicyFromCMDPPolicy(FiniteCAGPolicy):

    def __init__(self, cmdp_policy: CMDPPolicy, beta_0: Distribution):
        self._cmdp_policy: CMDPPolicy = cmdp_policy
        self._state_and_parameter_belief_space = cmdp_policy.S
        S, Theta = self._get_S_and_Theta_and_validate_state_and_parameter_belief_space()

        self._coordinator_action_space = cmdp_policy.A

        used_h_A, used_r_A = self._get_split_used_action_spaces_and_validate()

        super().__init__(S, used_h_A, used_r_A)
        self.Theta = Theta

        self.beta_0: Distribution = FiniteParameterDistribution(beta_0=beta_0, subset=frozenset(self.Theta))

    def _get_S_and_Theta_and_validate_state_and_parameter_belief_space(self) -> (FiniteSpace, set):
        S = set()
        Theta = set()
        for s_and_beta in self._state_and_parameter_belief_space:
            if not isinstance(s_and_beta, BeliefState):
                raise ValueError
            else:
                S.add(s_and_beta.s)
                for theta in s_and_beta.beta.support():
                    Theta.add(theta)
        return FiniteSpace(S), Theta

    def _get_split_used_action_spaces_and_validate(self) -> Tuple[FrozenSet[Action], FrozenSet[Action]]:
        used_r_A = {r_a for _, r_a in self._coordinator_action_space}
        plans = {p for p, _ in self._coordinator_action_space}
        used_h_A = set()
        for plan in plans:
            if not isinstance(plan, Plan):
                raise ValueError
            else:
                for val in plan.get_values():
                    used_h_A.add(val)
        return frozenset(used_h_A), frozenset(used_r_A)

    def _get_distribution(self, hist: Trajectory, theta) -> Distribution:
        bcmdp_state = self._get_bcmdp_state(hist)
        coordinator_action_dist = self._cmdp_policy(bcmdp_state)

        if coordinator_action_dist.is_almost_degenerate():
            assert isinstance(coordinator_action_dist, DiscreteDistribution)
            coordinator_action = coordinator_action_dist.get_mode()
            h_plan, r_a = coordinator_action
            h_a = h_plan(theta)
            return KroneckerDistribution(ActionPair(h_a, r_a))
        else:
            raise NotImplementedError("Not yet implemented for stochastic coordinator policies!")

    def _get_bcmdp_state(self, hist: Trajectory) -> BeliefState:
        betas: List[FiniteParameterDistribution] = [self.beta_0]
        for i in range(hist.t):
            bcmdp_state = BeliefState(hist.states[i], betas[i])
            coordinator_action_dist = self._cmdp_policy(bcmdp_state)

            if coordinator_action_dist.is_almost_degenerate():
                assert isinstance(coordinator_action_dist, DiscreteDistribution)
                coordinator_action = coordinator_action_dist.get_mode()
                h_plan = coordinator_action[0]
                h_a, _ = hist.actions[i]

                filter_func = (lambda possible_theta: h_plan(possible_theta) == h_a)
                next_beta = betas[i].get_collapsed_distribution_from_filter_func(filter_func)

                betas.append(next_beta)
            else:
                raise NotImplementedError("Not yet implemented for stochastic coordinator policies!")

        step_t_state = hist.states[hist.t]
        step_t_beta = betas[hist.t]

        step_t_bcmdp_state = BeliefState(step_t_state, step_t_beta)

        return step_t_bcmdp_state
