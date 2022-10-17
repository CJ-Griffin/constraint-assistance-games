from dataclasses import dataclass
from typing import Tuple

from src.formalisms.cmdp import CMDP
from src.formalisms.distributions import DiscreteDistribution
from src.formalisms.policy import MemorylessPolicy


@dataclass(frozen=True, eq=True)
class TStepTrajectory:
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
        assert len(self.states) == self.t + 1
        assert len(self.actions) == self.t
        assert len(self.rewards) == self.t
        assert (len(self.costs) == self.K)
        for k in range(self.K):
            assert len(self.costs[k]) == self.t

    def validate_for_cmdp(self, cmdp: CMDP):
        self.check_lengths()
        for s in self.states:
            if s not in cmdp.S:
                raise ValueError
        for a in self.actions:
            if a not in cmdp.A:
                raise ValueError

    def get_return(self):
        return self.get_discounted_sum(self.rewards)

    def get_kth_total_cost(self, k: int):
        return self.get_discounted_sum(self.costs[k])

    def get_discounted_sum(self, xs):
        return sum([xs[t] * (self.gamma ** t) for t in range(len(xs))])


def get_traj_dist(cmdp: CMDP, pol: MemorylessPolicy, prob_min_tol: float = 1e-9, timeout: int = 100):
    s_0_dist = cmdp.initial_state_dist

    def create_trajectory(t, states, actions, rewards, costs):
        return TStepTrajectory(t=t, states=states, actions=actions, rewards=rewards, costs=costs,
                               K=cmdp.K, gamma=cmdp.gamma)

    def get_next_traj_dist(prev_traj_dist: DiscreteDistribution):
        t = prev_traj_dist.sample().t

        next_step_traj_map = {}

        for prev_traj in prev_traj_dist.support():
            assert prev_traj.t == t
            prob_traj = prev_traj_dist.get_probability(prev_traj)
            last_state = prev_traj.states[-1]
            action_dist = pol(last_state)
            for poss_action in action_dist.support():
                prob_act = action_dist.get_probability(poss_action)
                next_state_dist = cmdp.T(last_state, poss_action)
                next_reward = cmdp.R(last_state, poss_action)
                next_costs = [
                    cmdp.C(k, last_state, poss_action) for k in range(cmdp.K)
                ]
                for poss_next_state in next_state_dist.support():
                    prob_next_state = next_state_dist.get_probability(poss_next_state)
                    next_traj = get_appended_traj(next_costs, next_reward, poss_action, poss_next_state, prev_traj, t)
                    assert next_traj not in next_step_traj_map
                    prob = prob_traj * prob_act * prob_next_state
                    next_step_traj_map[next_traj] = prob

        return DiscreteDistribution(next_step_traj_map)

    def get_appended_traj(next_costs, next_reward, poss_action, poss_next_state, prev_traj, t):
        next_traj = create_trajectory(
            t=t + 1,
            states=prev_traj.states + (poss_next_state,),
            actions=prev_traj.actions + (poss_action,),
            rewards=prev_traj.rewards + (next_reward,),
            costs=tuple(prev_traj.costs[k] + (next_costs[k],) for k in range(cmdp.K))
        )
        return next_traj

    zero_step_trajectory_dictionary = {
        create_trajectory(t=0, states=(s_0,), actions=tuple(), rewards=tuple(),
                          costs=tuple([tuple()] * cmdp.K)): 1.0
        for s_0 in s_0_dist.support()
    }
    zero_step_trajectory_dist = DiscreteDistribution(zero_step_trajectory_dictionary)

    traj_dists = [zero_step_trajectory_dist]
    unfinished_likely_trajectories = zero_step_trajectory_dictionary

    while len(unfinished_likely_trajectories) > 0:
        t_step_trajectories_dist = get_next_traj_dist(traj_dists[-1])
        unfinished_likely_trajectories = {
            traj: t_step_trajectories_dist.get_probability(traj)
            for traj in t_step_trajectories_dist.support()
            if not cmdp.is_sink(traj.states[-1])
            if t_step_trajectories_dist.get_probability(traj) > prob_min_tol
        }
        traj_dists.append(t_step_trajectories_dist)

        if len(traj_dists) > timeout:
            raise TimeoutError
    return traj_dists[-1]
