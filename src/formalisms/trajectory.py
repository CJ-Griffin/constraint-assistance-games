from dataclasses import dataclass
from typing import Tuple, FrozenSet, Callable

from src.formalisms.abstract_decision_processes import DecisionProcess
from src.formalisms.primitives import State, Action, Space
from src.reductions.cag_to_bcmdp import BeliefState
from src.renderer import render


@dataclass(frozen=True, eq=True)
class Trajectory:
    t: int
    states: Tuple[State, ...]
    actions: Tuple[Action, ...]

    def __post_init__(self):
        self.check_lengths()
        for s in self.states:
            if not isinstance(s, State):
                raise TypeError
        for a in self.actions:
            if not isinstance(a, Action):
                raise TypeError

    def check_lengths(self):
        assert len(self.states) == self.t + 1
        assert len(self.actions) == self.t

    def validate_for_process(self, process: DecisionProcess):
        self.check_lengths()
        for s in self.states:
            if s not in process.S:
                raise ValueError
        for a in self.actions:
            if a not in process.A:
                raise ValueError

    def get_whether_states_in_S(self, S: Space):
        return all([s in S for s in self.states])

    def get_whether_actions_in_A(self, A: FrozenSet[Action]):
        return all([a in A for a in self.actions])

    def get_next_trajectory(self, s: State, a: Action):
        return Trajectory(
            t=self.t + 1,
            states=self.states + (s,),
            actions=self.actions + (a,)
        )


@dataclass(frozen=True, eq=True)
class RewardfulTrajectory(Trajectory):
    t: int
    states: Tuple[State]
    actions: Tuple[Action, ...]
    rewards: Tuple[float, ...]
    K: int
    costs: Tuple[Tuple[float, ...], ...]
    gamma: float

    def __post_init__(self):
        self.check_lengths()

    def check_lengths(self):
        super(RewardfulTrajectory, self).check_lengths()

        assert len(self.rewards) == self.t
        assert (len(self.costs) == self.K)
        for k in range(self.K):
            assert len(self.costs[k]) == self.t

    def get_truncated_at_sink(self, sink_function: Callable[[State], bool]):
        for t in range(self.t):
            if sink_function(self.states[t]):
                return RewardfulTrajectory(
                    t=t,
                    states=tuple(self.states[0:t + 1]),
                    actions=tuple(self.actions[0:t]),
                    rewards=tuple(self.rewards[0:t]),
                    costs=tuple(tuple(cs[0:t]) for cs in self.costs),
                    K=self.K,
                    gamma=self.gamma)

    def get_return(self):
        return self.get_discounted_sum(self.rewards)

    def get_kth_total_cost(self, k: int):
        return self.get_discounted_sum(self.costs[k])

    def get_discounted_sum(self, xs):
        return sum([xs[t] * (self.gamma ** t) for t in range(len(xs))])

    def render(self) -> str:
        from tabulate import tabulate
        if not isinstance(self.states[0], BeliefState):
            def get_row(t):
                t_s_a = [t, self.states[t], self.actions[t], self.rewards[t]]
                costs = [self.costs[k][t] for k in range(self.K)]
                return t_s_a + costs

            rows = [
                get_row(t) for t in range(self.t)
            ]
            rows += [
                [self.t + 1, self.states[self.t]] + (["-"] * (self.K + 1))
            ]

            rows = map((lambda row: map(render, row)), rows)

            return tabulate(rows, headers=["t", "state", "action", "reward"] + [f"cost {k}" for k in range(self.K)])
        else:
            def get_row(t):
                t_s_a = [t, self.states[t].s, self.states[t].beta,
                         self.actions[t].a0, self.actions[t].a1, self.rewards[t]]
                costs = [self.costs[k][t] for k in range(self.K)]
                return t_s_a + costs

            rows = [
                get_row(t) for t in range(self.t)
            ]
            rows += [
                [self.t + 1, self.states[self.t].s, self.states[self.t].beta] + (["-"] * (self.K + 1))
            ]

            rows = [[render(cell) for cell in row] for row in rows]

            return tabulate(rows,
                            headers=["t", "b.s", "b.β", "λ", "ar", "reward"] + [f"cost {k}" for k in range(self.K)])


@dataclass(frozen=True, eq=True)
class CAGRewarfulTrajectory(RewardfulTrajectory):
    theta: object

    def render(self) -> str:
        return f"θ={render(self.theta)} \n" + RewardfulTrajectory.render(self)
