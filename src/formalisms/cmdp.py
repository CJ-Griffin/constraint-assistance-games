from src.formalisms.distributions import *
from abc import ABC, abstractmethod

from src.formalisms.spaces import Space, FiniteSpace
from tqdm import tqdm


class CMDP(ABC):
    S: Space = None
    A: set = None
    gamma: float = None
    K: int = None

    I: Distribution = None

    state = None

    def perform_checks(self):
        self.check_is_instantiated()
        self.check_I_is_valid()
        self.check_sinks()

    def check_sinks(self):
        sinks = {s for s in self.S if self.is_sink(s)}
        for s in sinks:
            for a in self.A:
                dist = self.T(s,a)
                p = dist.get_probability(s)
                if p < 1.0:
                    raise ValueError
                r = self.R(s,a)
                if r != 0.0:
                    raise ValueError

    @abstractmethod
    def T(self, s, a) -> Distribution:  # | None:
        pass

    @abstractmethod
    def R(self, s, a) -> float:
        pass

    @abstractmethod
    def C(self, k: int, s, a) -> float:
        assert k < self.K, f"k={k} is invalid, there are only K={self.K} cost functions"
        raise NotImplementedError

    @abstractmethod
    def c(self, k: int) -> float:
        # this should be
        # assert k < self.K, f"k={k} is invalid, there are only K={self.K} cost functions"
        raise NotImplementedError

    @abstractmethod
    def is_sink(self, s) -> bool:
        # this should be
        # assert s in self.S, f"s={s} is not in S={self.S}"
        raise NotImplementedError

    def check_is_instantiated(self):
        components = [
            self.S,
            self.A,
            self.I,
            self.gamma,
            self.K,
        ]
        if None in components:
            raise ValueError("Something hasn't been instantiated!")

    def check_I_is_valid(self):
        for s in self.I.support():
            if s not in self.S:
                raise ValueError(f"state s={s} is s.t. I(s) = "
                                 f"{self.I.get_probability(s)} but s is not in self.S={self.S}")
        return True

    def render_state_as_string(self, s) -> str:
        return str(s)


class FiniteCMDP(CMDP, ABC):
    S: FiniteSpace = None

    transition_matrix: np.array = None
    reward_matrix: np.array = None
    cost_matrix: np.array = None
    start_state_matrix: np.array = None
    state_to_ind_map: dict = None
    action_to_ind_map: dict = None

    states: list = None
    actions: list = None

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
        self.states = list(self.S)
        self.state_to_ind_map = {
            self.states[i]: i for i in range(len(self.states))
        }

        self.actions = list(self.A)
        self.action_to_ind_map = {
            self.actions[i]: i for i in range(len(self.actions))
        }

        self.reward_matrix = np.zeros((self.n_states, self.n_actions))
        self.transition_matrix = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.cost_matrix = np.zeros((self.K, self.n_states, self.n_actions))
        self.start_state_matrix = np.zeros(self.n_states)

        sm = self.state_to_ind_map
        am = self.action_to_ind_map
        for s in tqdm(self.S):
            self.start_state_matrix[sm[s]] = self.I.get_probability(s)
            if self.is_sink(s):
                # If the state is a sinK, we should self-loop!
                self.reward_matrix[sm[s], :] = 0
                self.transition_matrix[sm[s], :, :] = 0
                self.transition_matrix[sm[s], :, sm[s]] = 1.0
                self.cost_matrix[:, sm[s], :] = 0.0
            else:
                for a in self.A:
                    self.reward_matrix[sm[s], am[a]] = self.R(s, a)
                    dist = self.T(s, a)

                    for sp in dist.support():
                        s_ind = sm[s]
                        a_ind = am[a]
                        try:
                            sp_ind = sm[sp]
                        except KeyError as e:
                            if sp not in self.states:
                                close1 = [x for x in self.states if x[0] == sp[0]]
                                close2 = [x for x in self.states if x[1] == sp[1]]
                                close12 = [
                                    x for x in close1
                                    if tuple(x[1].support()) == tuple(sp[1].support())
                                ]
                                if len(close12) == 1:
                                    s_x, beta_x = close12[0]
                                    s_y, beta_y = sp
                                    beta_x.__eq__(beta_y)
                                raise ValueError(sp, "is not a state!")
                            else:
                                raise e
                        self.transition_matrix[s_ind, a_ind, sp_ind] = dist.get_probability(sp)

                    for k in range(self.K):
                        self.cost_matrix[k, sm[s], am[a]] = self.C(k, s, a)

    def validate(self):
        self.perform_checks()
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

    @staticmethod
    def is_stochastic_on_nth_dim(arr: np.ndarray, n: int):
        collapsed = arr.sum(axis=n)
        bools = collapsed == 1.0
        return bools.all()
