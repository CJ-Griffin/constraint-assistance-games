# from src.appr_grid_cag import ApprenticeshipStaticGridCAG
import datetime
import os

from src.env_wrapper import EnvWrapper
from src.formalisms.abstract_decision_processes import CAG, CMDP
from src.formalisms.policy import CMDPPolicy, FiniteCAGPolicy
from src.formalisms.trajectory import Trajectory
from src.get_traj_dist import get_traj_dist
from src.renderer import render
from src.utils import open_debug, get_path_relative_to_root, colors


def write_to_html(st, path):
    st = colors.term_to_html(st)
    start = f"""
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<!-- This file was created with the aha Ansi HTML Adapter. https://github.com/theZiz/aha -->
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="application/xml+xhtml; charset=UTF-8"/>
<title>stdin</title>
</head>
<body style="background-color:{colors.html.background_hex}; font-family: monospace; color: {colors.html.white_hex};">
<pre>
    """
    end = """
</pre>
</body>
</html>
"""
    with open_debug(path, "a+") as file:
        file.write(start + st + end)


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


def explore_CMDP_solution_extionsionally(policy: CMDPPolicy, solution_details: dict, supress_print: bool = False):
    soms = solution_details["state_occupancy_measures"]
    reached_states = [s for s in soms.keys() if soms[s] > 0]

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


def explore_CAG_policy_with_env_wrapper(policy: FiniteCAGPolicy,
                                        cag: CAG,
                                        should_render: bool = False,
                                        max_runs: int = None):
    thetas = list(cag.Theta)
    if max_runs is not None and max_runs < len(thetas):
        thetas = thetas[:max_runs]
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
        if should_render:
            full_traj = env.log.get_traj()
            print(render(full_traj))
