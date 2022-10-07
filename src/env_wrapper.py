import copy
from dataclasses import dataclass

from src.formalisms.mdp import MDP
from src.formalisms.cag import CAG
from gym import Env

from src.formalisms.cmdp import CMDP
from src.formalisms.cpomdp import CPOMDP


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


class EnvCAG(Env):

    def __init__(self, cag: CAG):
        self.cag = cag
        self.state = None
        self.theta = None
        self.t = 0

        self.reward_hist = []

        self.cost_totals = [0.0] * self.cag.K

        self.log = None
        self.log_archive = []

    def step(self, action_pair):
        a_h, a_r = action_pair
        s_t = copy.deepcopy(self.state)
        s_tp1_dist = self.cag.T(s_t, a_h, a_r)
        s_tp1 = s_tp1_dist.sample()
        obs = s_tp1

        r = self.cag.R(s_t, a_h, a_r)

        done = self.cag.is_sink(s_tp1)

        cur_costs = [self.cag.C(k, self.theta, s_t, a_h, a_r)
                     for k in range(self.cag.K)]
        # for k in range(self.cag.K):
        #     costs.append(self.cag.C(k, self.theta, s_t, a_h, a_r, s_tp1))
        # self.cost_totals[k] += (self.cag.gamma ** self.t) * costs[k]

        info = {"cur_costs": cur_costs}
        self.log.rewards.append(r)
        for k in range(self.cag.K):
            self.log.costs[k].append(cur_costs[k])

        self.t += 1
        self.state = s_tp1

        return obs, r, done, info

    def reset(self):
        self.state, self.theta = self.cag.I.sample()
        # self.cost_totals = [0.0] * self.cag.K
        self.t = 0
        if self.log is not None:
            self.log_archive.append(self.log)
        self.log = Log(
            s_0=self.state,
            theta=self.theta,
            gamma=self.cag.gamma,
            K=self.cag.K,
            budgets=[self.cag.c(k) for k in range(self.cag.K)]
        )
        return self.state

    def render(self, mode="human"):
        cost_str = ""
        # for k, cost_total in enumerate(self.cost_totals):
        #     cost_str += f"cost k={k} = {cost_total} of {self.cag.c(k)}\n"
        rend_str = f"""

        
===== State at t={self.t} =====
{self.cag.render_state_as_string(self.state)}
~~~~~ ------------ ~~~~~
reward history = {self.reward_hist}
{cost_str}
theta={self.theta}
===== ------------ =====
        """
        print(rend_str)


class EnvCPOMDP(Env):

    def __init__(self, cpomdp: CPOMDP):
        self.cpomdp = cpomdp
        self.state = None
        self.t = 0

        self.reward_hist = []

        self.cost_totals = [0.0] * self.cpomdp.K

    def step(self, a):
        s_t = copy.deepcopy(self.state)
        s_tp1_dist = self.cpomdp.T(s_t, a)
        s_tp1 = s_tp1_dist.sample()

        r = self.cpomdp.R(s_t, a)
        self.reward_hist.append(r)

        done = self.cpomdp.is_sink(s_tp1)

        obs_dist = self.cpomdp.O(a, s_tp1)
        obs = obs_dist.sample()

        costs = []
        for k in range(self.cpomdp.K):
            costs.append(self.cpomdp.C(k, s_t, a))
            self.cost_totals[k] += costs[k]

        info = {"costs": costs}

        self.t += 1
        self.state = s_tp1

        return obs, r, done, info

    def reset(self):
        self.state = self.cpomdp.b_0.sample()
        self.cost_totals = [0.0] * self.cpomdp.K
        self.t = 0
        return None

    def render(self, mode="human"):
        cost_str = ""
        for k, cost_total in enumerate(self.cost_totals):
            cost_str += f"cost k={k} = {cost_total} of {self.cpomdp.c(k)}\n"
        rend_str = f"""

===== State at t={self.t} =====
{self.cpomdp.render_state_as_string(self.state)}
~~~~~ ------------ ~~~~~
reward history = {self.reward_hist}
{cost_str}
===== ------------ =====
        """
        print(rend_str)


class EnvCMDP(Env):

    def __init__(self, cmdp: CMDP):
        self.cmdp = cmdp
        self.state = None
        self.t = 0

        self.reward_hist = []

        # self.cost_totals = [0.0] * self.cmdp.K
        self.log: Log = None

    def step(self, a):
        s_t = copy.deepcopy(self.state)
        s_tp1_dist = self.cmdp.T(s_t, a)
        s_tp1 = s_tp1_dist.sample()

        r = self.cmdp.R(s_t, a)

        done = self.cmdp.is_sink(s_tp1)

        obs = s_tp1

        cur_costs = [self.cmdp.C(k, s_t, a)
                     for k in range(self.cmdp.K)]

        info = {"cur_costs": cur_costs}

        self.t += 1
        self.state = s_tp1

        self.log.rewards.append(r)
        self.log.action_history.append(a)
        for k in range(self.cmdp.K):
            self.log.costs[k].append(cur_costs[k])

        return obs, r, done, info

    def reset(self, state=None):
        if state == None:
            self.state = self.cmdp.I.sample()
        else:
            if state in self.cmdp.S:
                self.state = state
            else:
                raise ValueError
        # self.cost_totals = [0.0] * self.cmdp.K
        self.t = 0
        self.log = Log(
            s_0=self.state,
            theta=None,
            gamma=self.cmdp.gamma,
            budgets=[self.cmdp.c(k) for k in range(self.cmdp.K)],
            K=self.cmdp.K
        )
        return self.state

    def render(self, mode="human"):
        cost_str = ""
        for k in range(self.cmdp.K):
            cost_total = self.log.total_kth_cost(k)
            cost_str += f"cost k={k} = {cost_total} of {self.cmdp.c(k)}\n"

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
{self.cmdp.render_state_as_string(self.state)}
~~~~~ ------------ ~~~~~
reward history = {self.log.rewards}
last action history = {last_action_string}
===== ------------ =====
        """
        print(rend_str)


class EnvMDP(Env):

    def __init__(self, mdp: MDP):
        self.mdp = mdp
        self.state = None
        self.t = 0

        self.reward_hist = []

        # self.cost_totals = [0.0] * self.cmdp.K
        self.log: Log = None

    def step(self, a):
        s_t = copy.deepcopy(self.state)
        s_tp1_dist = self.mdp.T(s_t, a)
        s_tp1 = s_tp1_dist.sample()

        r = self.mdp.R(s_t, a)

        self.log.rewards.append(r)
        self.log.action_history.append(a)

        done = self.mdp.is_sink(s_tp1)

        obs = s_tp1

        info = {}
        self.log.rewards.append(r)
        self.t += 1
        self.state = s_tp1

        return obs, r, done, info

    def reset(self):
        self.state = self.mdp.I.sample()
        # self.cost_totals = [0.0] * self.cmdp.K
        self.t = 0
        self.log = Log(
            s_0=self.state,
            theta=None,
            # rewards=[],
            gamma=self.mdp.gamma,
            # costs=[],
            budgets=[],
            K=0
        )
        return self.state

    def render(self, mode="human"):
        cost_str = ""
        rend_str = f"""

===== State at t={self.t} =====
{self.mdp.render_state_as_string(self.state)}
~~~~~ ------------ ~~~~~
reward history = {self.log.rewards}
last action history = {'NA' if len(self.log.action_history) == 0 else self.log.action_history[-1]}
{cost_str}
===== ------------ =====
        """
        print(rend_str)
