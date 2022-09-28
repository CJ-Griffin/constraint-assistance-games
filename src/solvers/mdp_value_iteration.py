from copy import copy

import numpy as np
from tqdm import tqdm

from src.formalisms import MDP, FiniteSpace


def get_value_function_and_policy_by_iteration(mdp: MDP,
                                               iter_limit=1000,
                                               show_tqdm: bool = False,
                                               tol: float = float(2 ** -8),
                                               restricted_action_map: dict = None) -> dict:
    S = mdp.S
    if isinstance(S, FiniteSpace):
        S = list(S)
        # S.sort()
        values = {s: 0.0 for s in S}
    else:
        raise ValueError

    A = mdp.A

    def get_ret_given_s_a_s_next(s, a, s_next) -> float:
        if mdp.is_sink(s_next):
            return mdp.R(s, a, s_next)
        else:
            return mdp.R(s, a, s_next) + (mdp.gamma * values[s_next])

    def get_Q_s_a_est(s, a) -> float:
        s_next_dist = mdp.T(s, a)
        terms = [
            s_next_dist.get_probability(s_next) * get_ret_given_s_a_s_next(s, a, s_next)
            for s_next in s_next_dist.support()
        ]
        return sum(terms)

    pbar = range(iter_limit)
    if show_tqdm:
        pbar = tqdm(pbar)

    max_diff_at_t = 1
    for t in pbar:
        max_diff_at_t = 0
        for s in S:
            if mdp.is_sink(s):
                values[s] = 0
            else:
                v = values[s]

                q_vals = [
                    get_Q_s_a_est(s, a)
                    for a in A
                ]
                values[s] = max(q_vals)
                max_diff_at_t = max(max_diff_at_t, np.absolute(v - values[s]))

        if max_diff_at_t <= tol:
            return values

        elif t == iter_limit - 1:
            raise TimeoutError
