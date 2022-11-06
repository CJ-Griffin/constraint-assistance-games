from src.appr_grid_cag import ApprenticeshipStaticGridCAG
from src.env_wrapper import EnvWrapper
from src.formalisms.abstract_decision_processes import CAG, CMDP
from src.formalisms.policy import CMDPPolicy, FiniteCAGPolicy
from src.formalisms.trajectory import Trajectory
from src.get_traj_dist import get_traj_dist
from src.renderer import render


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
        print()
        print()
        print(render(traj))
        print(f"Prob = {traj_dist.get_probability(traj)}")
        print()


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


def explore_CAG_with_keyboard_input(cag: CAG):
    for theta in cag.Theta:
        done = False
        env = EnvWrapper(cag)
        obs = env.reset(theta=theta)
        env.render()
        hist = Trajectory(t=0, states=(obs,), actions=tuple())

        if isinstance(cag, ApprenticeshipStaticGridCAG):
            control_scheme = {
                "8": (0, -1),
                "5": (0, 0),
                "2": (0, 1),
                "4": (-1, 0),
                "6": (1, 0),
                "w": (0, -1),
                "q": (0, 0),
                "s": (0, 1),
                "a": (-1, 0),
                "d": (1, 0)
            }
        else:
            raise NotImplementedError
            # human_action_list = list(cag.h_A)
            # control_scheme = {
            #     i: human_action_list[i]
            #     for i in range(len(human_action_list))
            # }
            # robot_action_list = list(cag.r_A)
            # for j, ra in enumerate(robot_action_list):
            #     control_scheme[len(human_action_list)+j] = ra
            # print(control_scheme)

        def get_action_pair():
            policy_map = dict()
            for poss_theta in cag.Theta:
                policy_map[poss_theta] = control_scheme[input(f"human {poss_theta}")]
            a_robot = control_scheme[input("robot")]
            return policy_map[env.theta], a_robot

        while not done:
            ah, ar = get_action_pair()
            obs, r, done, inf = env.step((ah, ar))
            hist = hist.get_next_trajectory(obs, (ah, ar))
            env.render()
