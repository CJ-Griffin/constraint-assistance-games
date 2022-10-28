import itertools
from abc import abstractmethod
from typing import List

from src.formalisms.plans import Plan
from src.formalisms.distributions import Distribution, UniformDiscreteDistribution, FiniteParameterDistribution, \
    KroneckerDistribution
from src.formalisms.spaces import Space, FiniteSpace, CountableSpace
from src.formalisms.trajectory import Trajectory


class CMDPPolicy:
    def __init__(self, S: Space, A: set):
        self.S: Space = S
        self.A: set = A

    @abstractmethod
    def _get_distribution(self, s) -> Distribution:
        pass

    def __call__(self, s) -> Distribution:
        if s not in self.S:
            raise ValueError
        else:
            return self._get_distribution(s)


class FiniteCMDPPolicy(CMDPPolicy):

    def __init__(self, S: Space, A: set, state_to_dist_map: dict, should_validate=True):
        if not isinstance(S, FiniteSpace):
            raise ValueError
        super().__init__(S, A)
        self._state_to_dist_map = state_to_dist_map

    def _get_distribution(self, s) -> Distribution:
        return self._state_to_dist_map[s]

    def validate(self):
        for s in self.S:
            if s not in self._state_to_dist_map:
                raise ValueError
            else:
                for a in self._state_to_dist_map[s].support():
                    if a not in self.A:
                        raise ValueError


class HistorySpaceIterator:
    def __init__(self, S, A):
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
    def __init__(self, S: FiniteSpace, A: set):
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
    def __init__(self, S: FiniteSpace, h_A: set, r_A: set):
        self.S: FiniteSpace = S
        self.A_joint_concrete: set = {(h_a, r_a) for h_a in h_A for r_a in r_A}
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

        h_A = used_h_A  # This might cause issues later!
        r_A = used_r_A
        super().__init__(S, h_A, r_A)
        self.Theta = Theta

        self.beta_0: Distribution = FiniteParameterDistribution(beta_0=beta_0, subset=frozenset(self.Theta))

    def _get_S_and_Theta_and_validate_state_and_parameter_belief_space(self) -> (FiniteSpace, set):
        S = set()
        Theta = set()
        for s_and_beta in self._state_and_parameter_belief_space:
            if not isinstance(s_and_beta, tuple) and len(s_and_beta) == 2:
                raise ValueError
            else:
                s, beta = s_and_beta
                S.add(s)
                for theta in beta.support():
                    Theta.add(theta)
        return FiniteSpace(S), Theta

    def _get_split_used_action_spaces_and_validate(self) -> (set, set):
        used_r_A = {r_a for _, r_a in self._coordinator_action_space}
        plans = {p for p, _ in self._coordinator_action_space}
        used_h_A = set()
        for plan in plans:
            if not isinstance(plan, Plan):
                raise ValueError
            else:
                for val in plan.get_values():
                    used_h_A.add(val)
        return used_h_A, used_r_A

    def _get_distribution(self, hist: Trajectory, theta) -> Distribution:
        bcmdp_state = self._get_bcmdp_state(hist)
        coordinator_action_dist = self._cmdp_policy(bcmdp_state)

        if coordinator_action_dist.is_degenerate():
            coordinator_action = coordinator_action_dist.sample()
            h_plan, r_a = coordinator_action
            h_a = h_plan(theta)
            return KroneckerDistribution((h_a, r_a))
        else:
            raise NotImplementedError("Not yet implemented for stochastic coordinator policies!")

    def _get_bcmdp_state(self, hist: Trajectory) -> object:
        betas: List[FiniteParameterDistribution] = [self.beta_0]
        for i in range(hist.t):
            bcmdp_state = (hist.states[i], betas[i])
            coordinator_action_dist = self._cmdp_policy(bcmdp_state)

            if coordinator_action_dist.is_degenerate():
                coordinator_action = coordinator_action_dist.sample()
                h_plan = coordinator_action[0]
                h_a, _ = hist.actions[i]

                filter_func = (lambda possible_theta: h_plan(possible_theta) == h_a)
                next_beta = betas[i].get_collapsed_distribution_from_filter_func(filter_func)

                betas.append(next_beta)
            else:
                raise NotImplementedError("Not yet implemented for stochastic coordinator policies!")

        step_t_state = hist.states[hist.t]
        step_t_beta = betas[hist.t]

        step_t_bcmdp_state = (step_t_state, step_t_beta)

        return step_t_bcmdp_state
