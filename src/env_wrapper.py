from dataclasses import dataclass

from gym import Env

from src.formalisms.abstract_decision_processes import DecisionProcess, CAG, CMDP, MDP
from src.formalisms.primitives import State, ActionPair, Action
from src.formalisms.trajectory import RewardfulTrajectory, CAGRewarfulTrajectory
from src.grid_world_cag import StaticGridWorldCAG
from src.grid_world_primitives import A_NORTH, A_NOOP, A_EAST, A_WEST, A_SOUTH
from src.renderer import render


@dataclass(frozen=False)
class Log:
    s_0: State
    theta: object  # Generic type
    gamma: float
    budgets: list
    K: int

    def __post_init__(self):
        self.states: list = []
        self.costs: list = [[] for k in range(self.K)]
        self.rewards: list = []
        self.actions: list = []

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

    def add_step(self, state: State, reward: float, action: Action, costs: list):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
        for k in range(self.K):
            self.costs[k].append(costs[k])

    def get_traj(self):
        if self.theta is None:
            return RewardfulTrajectory(
                t=len(self.actions),
                states=tuple(self.states),
                actions=tuple(self.actions),
                rewards=tuple(self.rewards),
                costs=tuple(tuple(cs) for cs in self.costs),
                K=self.K,
                gamma=self.gamma
            )
        else:
            return CAGRewarfulTrajectory(
                t=len(self.actions),
                states=tuple(self.states),
                actions=tuple(self.actions),
                rewards=tuple(self.rewards),
                costs=tuple(tuple(cs) for cs in self.costs),
                K=self.K,
                gamma=self.gamma,
                theta=self.theta
            )


class EnvWrapper(Env):
    def __init__(self,
                 process: DecisionProcess,
                 max_t_before_timeout: int = 200):
        self.process = process
        self.process.perform_checks()
        self.state = None
        self.theta = None
        self.t = 0
        self.max_t_before_timeout = max_t_before_timeout
        self.K = self.process.K

        self.reward_hist = []

        self.log: Log = None

    def get_cur_costs(self, s, a):
        if isinstance(self.process, MDP):
            return []
        elif isinstance(self.process, CMDP):
            return [self.process.C(k, s, a)
                    for k in range(self.K)]
        elif isinstance(self.process, CAG):
            a_h, a_r = a
            return [self.process.C(k, self.theta, s, a_h, a_r)
                    for k in range(self.K)]
        else:
            raise TypeError(self.process)

    def step(self, a):

        if isinstance(self.process, CAG) and not isinstance(a, ActionPair):
            raise TypeError

        if self.t > self.max_t_before_timeout:
            raise TimeoutError

        r = self.process.R(self.state, a)
        cur_costs = self.get_cur_costs(self.state, a)

        last_state = self.state

        next_state_dist = self.process.T(self.state, a)
        next_state = next_state_dist.sample()

        done = self.process.is_sink(next_state)
        obs = next_state
        self.state = next_state

        self.t += 1
        self.log.add_step(last_state, r, a, cur_costs)
        if done:
            self.log.states.append(next_state)

        info = {"cur_costs": cur_costs}
        return obs, r, done, info

    def reset(self, state=None, theta=None):
        if isinstance(self.process, (MDP, CMDP)):
            assert theta is None
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
            if theta is not None:
                self.theta = theta
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
            cost_str += f"ΣC{k}: {cost_total} <?= {self.process.c(k)}\n"

        if len(self.log.actions) == 0:
            last_action_string = "NA"
        else:
            last_a = self.log.actions[-1]
            if type(last_a) == tuple:
                last_action_string = "(" + ", ".join([str(x) for x in last_a]) + ")"
            else:
                last_action_string = str(last_a)

        if isinstance(self.process, CAG):
            theta_str = f"theta={render(self.theta)}"
        else:
            theta_str = ""

        rend_str = f"""

===== State at t={self.t} =====
{render(self.state)}
{theta_str}
~~~~~ ------------ ~~~~~
reward history = {self.log.rewards}
last action history = {last_action_string}
{cost_str}
===== ------------ =====
        """
        print(rend_str)


def play_decision_process(dp, theta=None):
    if isinstance(dp, StaticGridWorldCAG):
        def get_action_pair():
            control_scheme = {
                "w": A_NORTH,
                "q": A_NOOP,
                "d": A_EAST,
                "a": A_WEST,
                "s": A_SOUTH
            }
            x = control_scheme[input("ha=")]
            y = control_scheme[input("ra=")]
            return ActionPair(x, y)

        get_action = get_action_pair
    else:
        raise NotImplementedError

    if theta is not None and not isinstance(dp, CAG):
        raise TypeError

    with EnvWrapper(dp) as env:
        done = False
        _ = env.reset(theta=theta)
        env.render()
        while not done:
            a = get_action()
            _, _, done, _ = env.step(a)
            env.render()
