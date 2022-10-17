from copy import copy

import matplotlib.pyplot as plt
from tqdm import tqdm

from src.env_wrapper import EnvWrapper
from src.example_environments.randomised_cags_and_cmdps import RandJointPolicy, RandomisedCAG
from src.formalisms.cag import CAG
from src.formalisms.cag_to_bcmdp import CAGtoBCMDP


def run_jp_on_cag(jp: RandJointPolicy, cag: CAG):
    with EnvWrapper(cag) as env:
        s_0 = env.reset()
        hist = tuple()
        done = False
        while not done:
            h_lambda, r_a = jp.get_action_pair(hist)
            h_a = h_lambda(env.theta)
            s, r, done, inf = env.step((h_a, r_a))
        log = env.log
    return log


def run_jp_on_cag_reduced_to_cmdp(jp: RandJointPolicy, cag: CAG):
    cmdp = CAGtoBCMDP(copy(cag))
    with EnvWrapper(cmdp) as env:
        s_0 = env.reset()
        hist = tuple()
        done = False
        while not done:
            a_pair = jp.get_action_pair(hist)
            s, r, done, inf = env.step(a_pair)
        log = env.log
    return log


def test_random_cag():
    num_cags = 3
    num_agents = 3
    sample_size = 10
    rcags = [RandomisedCAG() for i in range(num_cags)]
    example_rcag = RandomisedCAG()
    agents = [RandJointPolicy(example_rcag) for i in range(num_agents)]

    import matplotlib.pyplot as plt

    f, axes = plt.subplots(num_cags, num_agents)

    iterator = tqdm([
        (i, j)
        for i in range(len(rcags))
        for j in range(len(agents))
    ])

    for i, j in iterator:
        rcag = rcags[i]
        jp = agents[j]
        ax = axes[i, j]
        specific_cmdp_logs = [run_jp_on_cag_reduced_to_cmdp(jp, rcag) for i in range(sample_size)]

        specific_cag_logs = [run_jp_on_cag(jp, rcag) for i in range(sample_size)]

        plot_spreads(ax, specific_cag_logs, specific_cmdp_logs)
        if j != 0:
            ax.set_yticks([])
        if i != len(rcags) - 1:
            ax.set_xticks([])

    plt.show()


def plot_spreads(ax: plt.Axes,
                 specific_cag_logs: list,
                 specific_cmdp_logs: list,
                 y_offset: float = 0.3,
                 marker_size: int = 30):
    K = specific_cag_logs[0].K
    cag_data = {
        "returns": [log.total_return() for log in specific_cag_logs]
    }
    for k in range(K):
        cag_data[f"cost {k}"] = [log.total_kth_cost(k) for log in specific_cag_logs]
    cmpd_data = {
        "returns": [log.total_return() for log in specific_cmdp_logs]
    }
    for k in range(K):
        cmpd_data[f"cost {k}"] = [log.total_kth_cost(k) for log in specific_cmdp_logs]

    # Create labels for graph
    cag_lab_ys = range(-1, K)
    cag_labs = ["CAG Return"] + [f"CAG Cost {k}" for k in range(K)]
    cmdp_lab_ys = [y + y_offset for y in range(-1, K)]
    cmdp_labs = ["CMDP Return"] + [f"CMDP Cost {k}" for k in range(K)]
    lab_ys = [val for pair in zip(cag_lab_ys, cmdp_lab_ys) for val in pair]
    labs = [val for pair in zip(cag_labs, cmdp_labs) for val in pair]
    ax.set_yticks(lab_ys, labs)
    ax.set_ylim(-1.5, K - 0.5)
    ax.plot([1.0, 1.0], [-0.5, K - 0.5], c="red", label="Budget", alpha=0.2)
    ax.set_xlim(0.5, 2.0)

    # Plot CAG numbers
    def mean(xs: list) -> float:
        return sum(xs) / len(xs)

    def plot_list(xs, y, c):
        ax.scatter(xs, [y for _ in range(len(xs))],
                   linestyle='None', marker="|", alpha=0.01, s=marker_size, c=c)
        ax.scatter([mean(xs)], [y],
                   linestyle='None', marker="|", alpha=1.0, s=marker_size * 5, c=c)

    plot_list(cag_data["returns"], -1, "green")
    for k in range(K):
        plot_list(cag_data[f"cost {k}"], k, c="red")
    # Plot CMDP numbers
    plot_list(cmpd_data["returns"], -1 + y_offset, c="purple")
    for k in range(K):
        plot_list(cmpd_data[f"cost {k}"], k + y_offset, c="orange")


if __name__ == "__main__":
    test_random_cag()
