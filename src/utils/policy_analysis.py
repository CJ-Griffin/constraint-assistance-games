# from src.appr_grid_cag import ApprenticeshipStaticGridCAG
import datetime
import os

from src.gym_env_wrapper import EnvWrapper
from src.formalisms.abstract_decision_processes import CAG, CMDP
from src.formalisms.policy import CMDPPolicy, FiniteCAGPolicy, FinitePolicyForFixedCMDP
from src.formalisms.trajectory import Trajectory
from src.utils.get_traj_dist import get_traj_dist
from src.abstract_gridworlds.grid_world_primitives import StaticGridState
from src.utils.renderer import render
from src.utils.utils import get_path_relative_to_root, write_to_html


def explore_CMDP_solution_with_trajectories(policy: CMDPPolicy,
                                            cmdp: CMDP,
                                            tol_min_prob: float = 1e-6,
                                            should_out_to_html: bool = True):
    traj_dist = get_traj_dist(
        cmdp=cmdp,
        pol=policy
    )

    fetch_prob = (lambda tr: traj_dist.get_probability(tr))
    filter_func = (lambda tr: traj_dist.get_probability(tr) > tol_min_prob)
    filtered_trajs = filter(filter_func, traj_dist.support())
    sorted_trajs = sorted(filtered_trajs, key=fetch_prob, reverse=True)
    out_str = ""
    for traj in sorted_trajs:
        out_str += "\n\n" + render(traj) + f"Prob = {traj_dist.get_probability(traj)}" + "\n"
    if not should_out_to_html:
        print(out_str)
    else:
        dt = datetime.datetime.now().strftime("%m_%d_%Y_%H%M%S")
        if hasattr(cmdp, "cag"):
            fn = "results" + os.sep + dt + "_bcmp_" + cmdp.cag.__class__.__name__ + ".html"
        else:
            fn = "results" + os.sep + dt + "_" + cmdp.__class__.__name__ + ".html"
        path = get_path_relative_to_root(fn)
        write_to_html(out_str, path)


def explore_CMDP_solution_extionsionally(policy: FinitePolicyForFixedCMDP,
                                         solution_details: dict,
                                         supress_print: bool = False):
    soms = solution_details["occupancy_measure_matrix"].sum(axis=1)
    reached_states = [
        s
        for s in policy.S
        if soms[policy.cmdp.state_to_ind_map[s]] > 0
    ]

    if supress_print:
        mprint = (lambda *x: None)
    else:
        mprint = print

    mprint("=" * 100)

    for state in reached_states:
        mprint()
        mprint("STATE:", render(state))
        s_ind = policy.cmdp.state_to_ind_map[state]
        mprint("STATE OCC. MEASURE:", render(soms[s_ind]))
        mprint("POLICY:", render(policy(state)))
        mprint()

    mprint(f"Value = {solution_details['objective_value']}")
    c_val_dict = solution_details["constraint_vals"]
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


def explore_CAG_policy_with_env_wrapper(policy: FiniteCAGPolicy,
                                        cag: CAG,
                                        should_render: bool = False,
                                        should_write_to_html: bool = False,
                                        max_runs: int = None):
    thetas = list(cag.Theta)
    if max_runs is not None and max_runs < len(thetas):
        thetas = thetas[:max_runs]
    trajs = []
    for theta in thetas:
        done = False
        env = EnvWrapper(cag)
        obs = env.reset(theta=theta)
        # if should_render:
        #     env.render()
        hist = Trajectory(t=0, states=(obs,), actions=tuple())
        while not done:
            a_dist = policy(hist, env.theta)
            a = a_dist.sample()
            obs, r, done, inf = env.step(a)
            hist = hist.get_next_trajectory(obs, a)
            # if should_render:
            # env.render()
        trajs.append(env.cur_traj)

    assert not (should_write_to_html and should_render)
    if should_render:
        for traj in trajs:
            print(render(traj))
    elif should_write_to_html:
        dt = datetime.datetime.now().strftime("%m_%d_%Y_%H%M%S")
        fn = "results" + os.sep + dt + "_" + cag.__class__.__name__ + ".html"
        path = get_path_relative_to_root(fn)

        st = ""
        if isinstance(trajs[0].states[0], StaticGridState):
            st += StaticGridState.get_legend_str() + "\n\n"

        for traj in trajs:
            st += render(traj) + "\n\n"

        write_to_html(st, path)
