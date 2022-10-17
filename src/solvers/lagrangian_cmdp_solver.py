import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.formalisms.cmdp import CMDP
from src.formalisms.lagrangian_cmdp_to_mdp import Lagrangian_CMDP_to_MDP
from src.solvers.mdp_value_iteration import get_value_function_and_policy_by_iteration


class FunctionWithMemory:
    def __init__(self, f):
        self.mem = dict()
        self.f = f

    def __call__(self, x):
        if x in self.mem:
            return self.mem[x]
        else:
            y = self.f(x)
            self.mem[x] = y
            return y


# Adapted from https://stackoverflow.com/questions/23042925/how-to-minimise-integer-function-thats-known-to-be-u-shaped
def find_minima_of_convex_f(f,
                            init_step_func=(lambda x: x * 2),
                            init_ub: float = float(2 ** -8),
                            absolute_precision: float = 1.e-3,
                            max_t: int = int(1000)):
    """
    Assumes f is convex and has its mimim(a/um) in the range [0, inf)
    :param init_ub:
    :param init_step_func:
    :param f:
    :return:
    """
    # test_xs = [x / 10.0 for x in range(100)]
    # plt.plot(test_xs, [f(x) for x in test_xs], alpha=0.2)
    # Invariant: x_l <= x_min <= x_u
    xs = [0.0, 0.0, init_ub]
    done = False
    while not done:
        if f(xs[-2]) <= f(xs[-1]):
            done = True
        else:
            xs.append(init_step_func(xs[-1]))

    left = xs[-3]
    right = xs[-1]

    for i in tqdm(range(0, max_t)):
        # while not done:
        if abs(right - left) < absolute_precision:
            break

        leftThird = (2 * left + right) / 3
        rightThird = (left + 2 * right) / 3

        if f(leftThird) <= f(rightThird):
            right = rightThird
            xs.append(right)
        else:
            left = leftThird
            xs.append(left)

    estimate = (left + right) / 2

    return estimate


def naive_lagrangian_cmdp_solver(cmdp: CMDP,
                                 mdp_solver=get_value_function_and_policy_by_iteration,
                                 show_results: bool = False,
                                 max_t: int = 1000):
    K = cmdp.K

    if K != 1:
        raise NotImplementedError

    def compute_d(lm: np.array) -> float:
        vf = mdp_solver(Lagrangian_CMDP_to_MDP(cmdp, lm))
        init_dist = cmdp.initial_state_dist
        value = sum([
            init_dist.get_probability(s_0) * vf[s_0]
            for s_0 in init_dist.support()
        ])
        cs = np.array([cmdp.c(k) for k in range(K)])
        prod = np.dot(cs, lm)
        return value + prod

    f = FunctionWithMemory(compute_d)
    lm_min = find_minima_of_convex_f(f, max_t=max_t)

    pairs = list(f.mem.items())
    print("\n".join([str(pair) for pair in pairs]))

    # This is kind of wasteful, but helps with generality
    value_function = mdp_solver(Lagrangian_CMDP_to_MDP(cmdp, lm_min))
    print()
    print(value_function)

    print()
    print(lm_min)
    print(f(lm_min))

    xs = list(f.mem.keys())
    if show_results:
        plt.scatter([lm_min], [f(lm_min)], s=300, marker="+", c="teal")
        plt.plot(xs, [f(x) for x in xs], "rx", alpha=0.3, linestyle='None')
        plt.show()

    return lm_min, value_function
