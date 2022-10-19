from dataclasses import dataclass
from typing import Tuple

from src.formalisms.abstract_process import AbstractProcess
from src.formalisms.spaces import Space


@dataclass(frozen=True, eq=True)
class Trajectory:
    t: int
    states: tuple
    actions: tuple

    def __post_init__(self):
        self.check_lengths()

    def check_lengths(self):
        assert len(self.states) == self.t + 1
        assert len(self.actions) == self.t

    def validate_for_process(self, process: AbstractProcess):
        self.check_lengths()
        for s in self.states:
            if s not in process.S:
                raise ValueError
        for a in self.actions:
            if a not in process.A:
                raise ValueError

    def get_whether_states_in_S(self, S: Space):
        return all([s in S for s in self.states])

    def get_whether_actions_in_A(self, A: set):
        return all([a in A for a in self.actions])

    def get_next_trajectory(self, s, a):
        return Trajectory(
            t=self.t + 1,
            states=self.states + (s,),
            actions=self.actions + (a,)
        )


@dataclass(frozen=True, eq=True)
class RewardfulTrajectory(Trajectory):
    t: int
    states: tuple
    actions: tuple
    rewards: Tuple[float]
    K: int
    costs: Tuple[Tuple[float]]
    gamma: float

    def __post_init__(self):
        self.check_lengths()

    def check_lengths(self):
        super(RewardfulTrajectory, self).check_lengths()

        assert len(self.rewards) == self.t
        assert (len(self.costs) == self.K)
        for k in range(self.K):
            assert len(self.costs[k]) == self.t

    def get_return(self):
        return self.get_discounted_sum(self.rewards)

    def get_kth_total_cost(self, k: int):
        return self.get_discounted_sum(self.costs[k])

    def get_discounted_sum(self, xs):
        return sum([xs[t] * (self.gamma ** t) for t in range(len(xs))])
