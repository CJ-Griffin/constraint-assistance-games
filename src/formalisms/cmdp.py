from tqdm import tqdm

from src.formalisms.abstract_process import AbstractProcess
from src.formalisms.distributions import *
from src.formalisms.spaces import FiniteSpace


class CMDP(AbstractProcess, ABC):
    initial_state_dist: Distribution = None

    @abstractmethod
    def C(self, k: int, s, a) -> float:
        raise NotImplementedError

    def check_init_dist_is_valid(self):
        for s in self.initial_state_dist.support():
            if s not in self.S:
                raise ValueError(f"state s={s} is s.t. I(s) = "
                                 f"{self.initial_state_dist.get_probability(s)} but s is not in self.S={self.S}")

    def test_cost_for_sinks(self):
        sinks = {s for s in self.S if self.is_sink(s)}
        for s in sinks:
            for a in self.A:
                for k in range(self.K):
                    if self.C(k, s, a) != 0.0:
                        raise ValueError("Cost should be 0 at a sink")

    def check_is_instantiated(self):
        if self.initial_state_dist is None:
            raise ValueError("init dist hasn't been instantiated!")
        super().check_is_instantiated()


class FiniteCMDP(CMDP, ABC):
    S: FiniteSpace = None

    transition_matrix: np.array = None
    reward_matrix: np.array = None
    cost_matrix: np.array = None
    start_state_matrix: np.array = None
    state_to_ind_map: dict = None
    action_to_ind_map: dict = None

    state_list: list = None
    action_list: list = None

    @property
    def n_states(self):
        return len(self.S)

    @property
    def n_actions(self):
        return len(self.A)

    @property
    def transition_probabilities(self) -> np.array:
        if self.transition_matrix is None:
            self.initialise_matrices()
        return self.transition_matrix

    @property
    def rewards(self) -> np.array:
        if self.reward_matrix is None:
            self.initialise_matrices()
        return self.reward_matrix

    @property
    def costs(self) -> np.array:
        if self.cost_matrix is None:
            self.initialise_matrices()
        return self.cost_matrix

    @property
    def start_state_probabilities(self) -> np.array:
        if self.start_state_matrix is None:
            self.initialise_matrices()
        return self.start_state_matrix

    def initialise_matrices(self):
        self.state_list = list(self.S)
        self.state_to_ind_map = {
            self.state_list[i]: i for i in range(len(self.state_list))
        }

        self.action_list = list(self.A)
        self.action_to_ind_map = {
            self.action_list[i]: i for i in range(len(self.action_list))
        }

        self.reward_matrix = np.zeros((self.n_states, self.n_actions))
        self.transition_matrix = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.cost_matrix = np.zeros((self.K, self.n_states, self.n_actions))
        self.start_state_matrix = np.zeros(self.n_states)

        sm = self.state_to_ind_map
        am = self.action_to_ind_map
        for s in tqdm(self.S):
            self.start_state_matrix[sm[s]] = self.initial_state_dist.get_probability(s)
            for a in self.A:
                self.reward_matrix[sm[s], am[a]] = self.R(s, a)
                dist = self.T(s, a)

                for sp in dist.support():
                    s_ind = sm[s]
                    a_ind = am[a]
                    sp_ind = sm[sp]
                    self.transition_matrix[s_ind, a_ind, sp_ind] = dist.get_probability(sp)

                for k in range(self.K):
                    self.cost_matrix[k, sm[s], am[a]] = self.C(k, s, a)

    def check_matrices(self):
        assert self.n_states is not None
        assert self.n_actions is not None

        assert self.S is not None
        assert self.A is not None
        assert self.rewards is not None
        assert self.transition_probabilities is not None
        assert self.start_state_probabilities is not None

        assert self.rewards.shape == (self.n_states, self.n_actions)
        assert self.transition_probabilities.shape == (self.n_states, self.n_actions, self.n_states)
        assert self.start_state_probabilities.shape == (self.n_states,)

        assert self.is_stochastic_on_nth_dim(self.transition_probabilities, 2)
        assert self.is_stochastic_on_nth_dim(self.start_state_probabilities, 0)
        self.perform_checks()
        self.stoch_check_if_matrices_match()

    @staticmethod
    def is_stochastic_on_nth_dim(arr: np.ndarray, n: int):
        collapsed = arr.sum(axis=n)
        bools = collapsed == 1.0
        return bools.all()

    def stoch_check_if_matrices_match(self, num_checks=100):
        num_checks = min([self.n_actions, self.n_states, num_checks])
        sm = self.state_to_ind_map
        am = self.action_to_ind_map

        s_ind_list = np.random.choice(self.n_states, size=num_checks, replace=False)
        a_ind_list = np.random.choice(self.n_actions, size=num_checks, replace=False)
        for i in range(num_checks):
            s = self.state_list[s_ind_list[i]]
            a = self.action_list[a_ind_list[i]]
            s_next_dist = self.T(s, a)
            i = sm[s]
            j = am[a]

            reward_from_R = self.R(s, a)
            reward_from_mat = self.reward_matrix[i, j]

            if reward_from_R != reward_from_mat:
                raise ValueError

            for s_next in self.state_list:
                prob_T = s_next_dist.get_probability(s_next)
                k = sm[s_next]
                prob_matrix = self.transition_matrix[i, j, k]
                if prob_T != prob_matrix:
                    raise ValueError
