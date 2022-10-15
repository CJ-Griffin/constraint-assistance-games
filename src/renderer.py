from src.formalisms.appr_grid_cag import ASGState
from src.formalisms.cag_to_bcmdp import Plan
from src.formalisms.distributions import DiscreteDistribution
from src.get_traj_dist import TStepTrajectory

_PRIMITIVES = (int, str, bool, float)
_BASIC_COMPOSITES = (list, tuple)


# TODO - convert this into a set of tree-like functions, and then one stringify function
def render(x: object) -> str:
    if isinstance(x, _PRIMITIVES):
        return str(x)
    elif isinstance(x, _BASIC_COMPOSITES):
        return _render_basic_composite(x)
    elif isinstance(x, DiscreteDistribution):
        return _render_discrete_distribution(x)
    elif isinstance(x, ASGState):
        return x.render()
    elif isinstance(x, Plan):
        return _render_plan(x)
    elif isinstance(x, TStepTrajectory):
        return _render_traj(x)
    else:
        raise NotImplementedError(x)


def _render_traj_tabularly(traj: TStepTrajectory) -> str:
    from tabulate import tabulate

    def get_row(t):
        t_s_a = [t, traj.states[t], traj.actions[t], traj.rewards[t]]
        costs = [traj.costs[k][t] for k in range(traj.K)]
        return t_s_a + costs

    rows = [
        get_row(t) for t in range(traj.t)
    ]
    rows += [
        [traj.t + 1, traj.states[traj.t]] + (["-"] * (traj.K + 1))
    ]

    rows = map((lambda row: map(render, row)), rows)

    return tabulate(rows, headers=["t", "state", "action", "reward"] + [f"cost {k}" for k in range(traj.K)])


def _rend_traj_old(traj: TStepTrajectory) -> str:
    s0 = _add_indents(render(traj.states[0]))
    rend_str = f"s0={s0}"
    for t in range(0, traj.t):
        a_t = traj.actions[t]
        r_t = traj.rewards[t]
        rend_str += f"\n a{t}={_add_indents(render(a_t))}"
        rend_str += f"\n r{t}={render(r_t)}"
        for k in range(traj.K):
            kth_c_t = traj.costs[k][t]
            rend_str += f"\n {k}th c{t}={render(kth_c_t)}"

        s_tp1 = traj.states[t]
        rend_str += f"\ns{t + 1}={_add_indents(render(s_tp1))}"
    return rend_str


def _render_traj(traj: TStepTrajectory, is_tabular: bool = True):
    if is_tabular:
        rend_str = _render_traj_tabularly(traj)
    else:
        rend_str = _rend_traj_old(traj)

    rend_str += f"\nReturn={traj.get_return()}"
    for k in range(traj.K):
        rend_str += f"\nTotal {k}th cost={traj.get_kth_total_cost(k)}"
    return rend_str


def _render_plan(plan: Plan) -> str:
    string = f"<_Plan"
    for key in plan.get_keys():
        string += f"\n :  {render(key)} -> {render(plan(key))}"
    string += "\n"
    return string


def _add_indents(s: str):
    return s.replace("\n", "\n    ")


def _render_discrete_distribution(dist: DiscreteDistribution) -> str:
    r_str = f"{type(dist).__name__}"
    sup = list(dist.support())
    if len(sup) > 1:
        for x in sup:
            xstr = render(x)
            p = dist.get_probability(x)
            r_str += f"\n |  {p: 4.3f} ~> {xstr}"
    else:
        r_str += f"(1~> {render(sup[0])})"
    return r_str


def _render_tuple(t: tuple):
    return "(" + ", ".join([render(y) for y in t]) + ")"


def _render_list(xs: list):
    return "[" + ", ".join([render(y) for y in xs]) + "]"


def _render_basic_composite(xs):
    if isinstance(xs, tuple):
        return _render_tuple(xs)
    elif isinstance(xs, list):
        return _render_list(xs)
    else:
        raise NotImplementedError
