from typing import TextIO

from src.env_wrapper import EnvWrapper
from src.formalisms.abstract_process import AbstractProcess
from src.formalisms.cag import CAG
from src.formalisms.cmdp import CMDP
from src.formalisms.distributions import Distribution
from src.formalisms.policy import CMDPPolicy, FiniteCAGPolicy
from src.formalisms.trajectory import Trajectory
from src.get_traj_dist import get_traj_dist
from src.renderer import render


def open_debug(file_name: str, *args, **kwargs) -> TextIO:
    try:
        file = open(file_name, *args, **kwargs)
    except FileNotFoundError as fnfe:
        import os
        cwd = os.getcwd()
        ls = os.listdir(cwd)
        raise fnfe
    return file


def explore_CMDP_solution_with_trajectories(policy: CMDPPolicy,
                                            cmdp: CMDP, tol_min_prob: float = 1e-6):
    traj_dist = get_traj_dist(
        cmdp=cmdp,
        pol=policy
    )

    fetch_prob = (lambda tr: traj_dist.get_probability(tr))
    filter_func = (lambda tr: traj_dist.get_probability(tr) > tol_min_prob)
    filtered_trajs = filter(filter_func, traj_dist.support())
    sorted_trajs = sorted(filtered_trajs, key=fetch_prob, reverse=True)
    for traj in sorted_trajs:
        print(render(traj))
        print(f"Prob = {traj_dist.get_probability(traj)}")
        print()


def explore_CMDP_solution_extionsionally(policy: CMDPPolicy, solution_details: dict, supress_print: bool = False):
    soms = solution_details["state_occupancy_measures"]
    reached_states = [s for s in soms.keys() if soms[s] > 0]
    reached_states.sort(key=(lambda x: str(x[1]) + str(x[0])))

    if supress_print:
        mprint = (lambda *x: None)
    else:
        mprint = print

    mprint("=" * 100)

    for state in reached_states:
        mprint()
        mprint("STATE:", render(state))
        mprint("STATE OCC. MEASURE:", render(soms[state]))
        mprint("POLICY:", render(policy(state)))
        mprint()

    mprint(f"Value = {solution_details['objective_value']}")
    c_val_dict = solution_details["constraint_values"]
    for constr_name in c_val_dict:
        mprint(f"{constr_name} => {c_val_dict[constr_name]}")


def explore_CMDP_policy_with_env_wrapper(policy: CMDPPolicy, cmdp: CMDP, should_render: bool = False):
    done = False
    env = EnvWrapper(cmdp)
    obs = env.reset()
    if should_render:
        env.render()
    while not done:
        a = policy(obs).sample()
        obs, r, done, inf = env.step(a)
        if should_render:
            env.render()


def explore_CAG_policy_with_env_wrapper(policy: FiniteCAGPolicy, cag: CAG, should_render: bool = False):
    done = False
    env = EnvWrapper(cag)
    obs = env.reset()
    if should_render:
        env.render()
    hist = Trajectory(t=0, states=(obs,), actions=tuple())
    while not done:
        a = policy(hist, env.theta).sample()
        obs, r, done, inf = env.step(a)
        hist = hist.get_next_trajectory(obs, a)
        if should_render:
            env.render()
