from gym import Env
from src.abstract_gridworlds.grid_world_cag import StaticGridWorldCAG
from src.formalisms.abstract_decision_processes import DecisionProcess, CAG, CMDP, MDP
from src.formalisms.primitives import ActionPair, Plan
from src.formalisms.trajectory import RewardfulTrajectory, CAGRewarfulTrajectory
from src.reductions.cag_to_bcmdp import CAGtoBCMDP
from src.utils.renderer import render


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
        self.cur_traj: RewardfulTrajectory = None

    def get_cur_costs(self, s, a):
        if isinstance(self.process, MDP):
            return tuple()
        elif isinstance(self.process, CMDP):
            return tuple(
                self.process.C(k, s, a)
                for k in range(self.K)
            )
        elif isinstance(self.process, CAG):
            a_h, a_r = a
            return tuple(
                self.process.C(k, self.theta, s, a_h, a_r)
                for k in range(self.K)
            )
        else:
            raise TypeError(self.process)

    def step(self, a):

        if isinstance(self.process, CAG) and not isinstance(a, ActionPair):
            raise TypeError

        if self.t > self.max_t_before_timeout:
            raise TimeoutError

        r = self.process.R(self.state, a)
        cur_costs = self.get_cur_costs(self.state, a)

        next_state_dist = self.process.T(self.state, a)
        next_state = next_state_dist.sample()

        done = self.process.is_sink(next_state)
        obs = next_state
        self.state = next_state

        self.t += 1
        self.cur_traj = self.cur_traj.get_next_rewardful_trajectory(s_next=next_state,
                                                                    a=a,
                                                                    r=r,
                                                                    cur_costs=cur_costs)

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
            self.cur_traj = RewardfulTrajectory(
                t=0,
                states=(self.state,),
                actions=tuple(),
                rewards=tuple(),
                K=self.K,
                costs=tuple(tuple() for k in range(self.process.K)),
                gamma=self.process.gamma,
                budgets=tuple(self.process.c(k) for k in range(self.K))
            )
        elif isinstance(self.process, CAG):
            if state is not None:
                raise NotImplementedError
            sample = self.process.initial_state_theta_dist.sample()
            if not isinstance(sample, tuple):
                raise TypeError("CAG I should be over S and Theta")
            self.state, self.theta = sample
            if theta is not None:
                self.theta = theta
            self.cur_traj = CAGRewarfulTrajectory(
                t=0,
                states=(self.state,),
                actions=tuple(),
                rewards=tuple(),
                K=self.K,
                costs=tuple(tuple() for k in range(self.process.K)),
                gamma=self.process.gamma,
                budgets=tuple(self.process.c(k) for k in range(self.K)),
                theta=self.theta
            )
        else:
            raise TypeError(self.process)

        self.t = 0
        return self.state

    def render(self, mode="human"):
        cost_str = ""
        for k in range(self.K):
            cost_total = self.cur_traj.get_kth_total_cost(k)
            cost_str += f"ΣC{k}: {cost_total} <?= {self.process.c(k)}\n"

        if len(self.cur_traj.actions) == 0:
            last_action_string = "NA"
        else:
            last_a = self.cur_traj.actions[-1]
            last_action_string = render(last_a)

        if isinstance(self.process, CAG):
            theta_str = f"theta={render(self.theta)}"
        else:
            theta_str = ""

        rend_str = f"""

===== State at t={self.t} =====
{render(self.state)}
{theta_str}
~~~~~ ------------ ~~~~~
reward history = {self.cur_traj.rewards}
last action history = {last_action_string}
{cost_str}
===== ------------ =====
        """
        print(rend_str)


def play_decision_process(dp, theta=None):
    if isinstance(dp, StaticGridWorldCAG):
        def get_action_pair():
            x = StaticGridWorldCAG.CONTROL_SCHEME[input("ha=")]
            y = StaticGridWorldCAG.CONTROL_SCHEME[input("ra=")]
            return ActionPair(x, y)

        get_action = get_action_pair
    elif isinstance(dp, CAGtoBCMDP) and isinstance(dp.cag, StaticGridWorldCAG):
        def get_coordinator_action():
            cs = StaticGridWorldCAG.CONTROL_SCHEME
            policy_map = dict()
            for poss_theta in dp.cag.Theta:
                policy_map[poss_theta] = cs[input(f"human {render(poss_theta)}")]
            a_robot = cs[input("robot")]
            return ActionPair(Plan(policy_map), a_robot)

        get_action = get_coordinator_action
    else:
        print()
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
