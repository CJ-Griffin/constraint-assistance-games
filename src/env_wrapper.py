from dataclasses import dataclass

from src.formalisms.abstract_process import AbstractProcess
from src.formalisms.mdp import MDP
from src.formalisms.cag import CAG
from gym import Env
from src.formalisms.cmdp import CMDP


@dataclass(frozen=False)
class Log:
    s_0: object  # Generic type
    theta: object  # Generic type
    gamma: float
    budgets: list
    K: int

    def __post_init__(self):
        self.costs: list = [[] for k in range(self.K)]
        self.rewards: list = []
        self.action_history: list = []

    def _calculate_discounted_sum(self, l: list):
        return sum([
            l[t] * (self.gamma ** t) for t in range(len(l))
        ])

    def total_return(self):
        return self._calculate_discounted_sum(self.rewards)

    def total_kth_cost(self, k: int):
        assert k < len(self.costs)
        return self._calculate_discounted_sum(self.costs[k])

    def convert_to_eval_string(self) -> str:
        try:
            str_lines = [""]
            ret = self.total_return()
            str_lines.append(f"return = {ret}")
            str_lines.append(f"costs  = ...")
            for k in range(len(self.costs)):
                total_kth_cost = self.total_kth_cost(k)
                budget = self.budgets[k]
                if total_kth_cost <= budget:
                    str_lines.append(f"  {k}| {total_kth_cost} <= {budget}")
                else:
                    str_lines.append(f"  {k}| {total_kth_cost} > {budget} EXCEEDED")
            str_lines.append("")
            return "\n".join(str_lines)
        except KeyError as e:
            # Try and except to allow debug break
            raise e

    def add_step(self, reward: float, action, costs: list):
        self.rewards.append(reward)
        self.action_history.append(action)
        for k in range(self.K):
            self.costs[k].append(costs[k])


class EnvWrapper(Env):
    def __init__(self, process: AbstractProcess, max_t_before_timeout: int = 100):
        self.process = process
        self.state = None
        self.theta = None
        self.t = 0
        self.max_t_before_timeout = max_t_before_timeout
        self.K = self.process.K

        self.reward_hist = []

        self.log: Log = None

    def get_cur_costs(self, s_t, a):
        if isinstance(self.process, MDP):
            return []
        elif isinstance(self.process, CMDP):
            return [self.process.C(k, s_t, a)
                    for k in range(self.K)]
        elif isinstance(self.process, CAG):
            a_h, a_r = a
            return [self.process.C(k, self.theta, s_t, a_h, a_r)
                    for k in range(self.K)]
        else:
            raise TypeError(self.process)

    def step(self, a):

        if isinstance(self.process, CAG) and not isinstance(a, tuple):
            raise TypeError

        if self.t > self.max_t_before_timeout:
            raise TimeoutError

        r = self.process.R(self.state, a)
        cur_costs = self.get_cur_costs(self.state, a)

        s_tp1_dist = self.process.T(self.state, a)
        s_tp1 = s_tp1_dist.sample()

        done = self.process.is_sink(s_tp1)
        obs = s_tp1
        self.state = s_tp1

        self.t += 1
        self.log.add_step(r, a, cur_costs)

        info = {"cur_costs": cur_costs}
        return obs, r, done, info

    def reset(self, state=None):
        if isinstance(self.process, (MDP, CMDP)):
            if state is None:
                self.state = self.process.initial_state_dist.sample()
            else:
                if state in self.process.S:
                    self.state = state
                else:
                    raise ValueError
        elif isinstance(self.process, CAG):
            if state is not None:
                raise NotImplementedError
            sample = self.process.initial_state_theta_dist.sample()
            if not isinstance(sample, tuple):
                raise TypeError("CAG I should be over S and Theta")
            self.state, self.theta = sample
        else:
            raise TypeError(self.process)

        self.t = 0
        self.log = Log(
            s_0=self.state,
            theta=self.theta,
            gamma=self.process.gamma,
            budgets=[self.process.c(k) for k in range(self.K)],
            K=self.K
        )
        return self.state

    def render(self, mode="human"):
        cost_str = ""
        for k in range(self.K):
            cost_total = self.log.total_kth_cost(k)
            cost_str += f"cost {k} of {self.K} = {cost_total} <?= {self.process.c(k)}\n"

        if len(self.log.action_history) == 0:
            last_action_string = "NA"
        else:
            last_a = self.log.action_history[-1]
            if type(last_a) == tuple:
                last_action_string = "(" + ", ".join([str(x) for x in last_a]) + ")"
            else:
                last_action_string = str(last_a)

        rend_str = f"""

===== State at t={self.t} =====
{self.process.render_state_as_string(self.state)}
~~~~~ ------------ ~~~~~
reward history = {self.log.rewards}
last action history = {last_action_string}
{cost_str}
===== ------------ =====
        """
        print(rend_str)
