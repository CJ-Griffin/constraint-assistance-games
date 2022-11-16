from dataclasses import dataclass
from typing import Tuple, FrozenSet, Callable

from src.formalisms.abstract_decision_processes import DecisionProcess
from src.formalisms.ecas_cags import EthicalContext
from src.formalisms.primitives import State, Action, Space, ActionPair, Plan
from src.reductions.cag_to_bcmdp import BeliefState
from src.utils.renderer import render
from src.utils.utils import colors


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

    def get_next_trajectory(self, s_next: State, a: Action):
        return Trajectory(
            t=self.t + 1,
            states=self.states + (s_next,),
            actions=self.actions + (a,)
        )


@dataclass(frozen=True, eq=True)
class RewardfulTrajectory(Trajectory):
    t: int
    states: Tuple[State, ...]
    actions: Tuple[Action, ...]
    rewards: Tuple[float, ...]
    K: int
    costs: Tuple[Tuple[float, ...], ...]
    gamma: float
    budgets: Tuple[float, ...]

    def __post_init__(self):
        self.check_lengths()

    def check_lengths(self):
        super(RewardfulTrajectory, self).check_lengths()

        assert len(self.rewards) == self.t
        assert (len(self.costs) == self.K)
        assert len(self.budgets) == self.K
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
                    gamma=self.gamma,
                    budgets=self.budgets
                )

    def get_return(self):
        return self.get_discounted_sum(self.rewards)

    def get_kth_total_cost(self, k: int):
        return self.get_discounted_sum(self.costs[k])

    def get_discounted_sum(self, xs):
        return sum([xs[t] * (self.gamma ** t) for t in range(len(xs))])

    def render(self) -> str:
        from tabulate import tabulate

        headers = ["t"]
        columns = [[render(t) for t in range(self.t + 1)]]
        if isinstance(self.states[0], BeliefState):
            headers += ["b.s", "b.β"]
            columns.append([render(b.s) for b in self.states])
            columns.append([render(b.beta) for b in self.states])
        else:
            headers.append("s")
            columns.append([render(s) for s in self.states])

        if isinstance(self.actions[0], ActionPair):
            if isinstance(self.actions[0].a0, Plan):
                headers += ["λ", "ar"]
            else:
                headers += ["ah", "ar"]
            columns.append([render(a.a0) for a in self.actions] + ["-"])
            columns.append([render(a.a1) for a in self.actions] + ["-"])
        else:
            headers.append("a")
            columns.append([render(a) for a in self.actions] + ["-"])

        headers.append("r")
        columns.append([render(r) for r in self.rewards] + [f"Σγᵗr={self.get_return():.2f}"])

        for k in range(self.K):
            headers.append(f"C{k} (c{k}={self.budgets[k]})")
            columns.append([render(cost) for cost in self.costs[k]] + [f"ΣγᵗC{k}={self.get_kth_total_cost(k):.2f}"])

        assert all(self.t + 1 == len(column) for column in columns)
        rows = [
            [columns[col_no][t] for col_no in range(len(columns))]
            for t in range(self.t + 1)
        ]

        return tabulate(rows, headers=headers) + "\n\n" + self.get_score_str() + "\n"

    def get_score_str(self):
        scores_str = ""

        triplets = [("R ", self.rewards, None)] + [(f"C{k}", self.costs[k], self.budgets[k]) for k in range(self.K)]
        for label, xs, budget in triplets:
            srt_st = f"\nΣ{label}= "
            if self.gamma >= 1.0:
                scores_str += srt_st + " + ".join(f"({self.gamma ** t * xs[t]:4.2f})" for t in range(len(xs)))
            else:
                scores_str += srt_st + " + ".join(f"(γᵗ *{xs[t]:5})" for t in range(len(xs)))
                scores_str += srt_st + " + ".join(f"({self.gamma ** t:1.2f}*{xs[t]:4})" for t in range(len(xs)))
                scores_str += srt_st + " + ".join(f"({self.gamma ** t * xs[t]:9.2f})" for t in range(len(xs)))
            if budget is None:
                scores_str += srt_st + str(f"{self.get_discounted_sum(xs):.4f}")
            else:
                assert isinstance(budget, float)
                total = self.get_discounted_sum(xs)
                scores_str += srt_st + str(f"{total:.4f}")
                if total > budget:
                    scores_str += colors.term.red(f"> {budget:3.1f}")
                else:
                    scores_str += colors.term.green(f"<={budget:3.1f}")
            scores_str += "\n"
        return scores_str

    def get_next_rewardful_trajectory(self, s_next: State, a: Action, r: float, cur_costs: Tuple[float, ...]):
        return RewardfulTrajectory(
            t=self.t + 1,
            states=self.states + (s_next,),
            actions=self.actions + (a,),
            rewards=self.rewards + (r,),
            K=self.K,
            costs=tuple(tuple(cs) + (c,) for cs, c in zip(self.costs, cur_costs)),
            gamma=self.gamma,
            budgets=self.budgets
        )


@dataclass(frozen=True, eq=True)
class CAGRewarfulTrajectory(RewardfulTrajectory):
    theta: object

    def render(self) -> str:
        if isinstance(self.theta, EthicalContext):
            start_char = "ℰ*"
        else:
            start_char = "θ*"
        return f"{start_char}={render(self.theta)} \n" + RewardfulTrajectory.render(self)

    def get_next_rewardful_trajectory(self, s_next: State, a: Action, r: float, cur_costs: Tuple[float, ...]):
        return CAGRewarfulTrajectory(
            t=self.t + 1,
            states=self.states + (s_next,),
            actions=self.actions + (a,),
            rewards=self.rewards + (r,),
            K=self.K,
            costs=tuple(tuple(cs) + (c,) for cs, c in zip(self.costs, cur_costs)),
            gamma=self.gamma,
            budgets=self.budgets,
            theta=self.theta
        )
