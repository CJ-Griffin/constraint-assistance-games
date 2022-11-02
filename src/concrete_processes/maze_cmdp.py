from dataclasses import dataclass

import numpy as np

from src.formalisms.distributions import Distribution, KroneckerDistribution
from src.formalisms.finite_processes import FiniteCMDP
from src.formalisms.primitives import State, IntAction, FiniteSpace


@dataclass(frozen=True, eq=True)
class XYState(State):
    x: int
    y: int

    def render(self) -> str:
        grid = np.array([
            [".", ".", "."],
            ["@", "@", "."],
            ["*", ".", "."],
        ])
        grid[self.y, self.x] = "p"

        grid_str = "\n".join([" ".join(grid[y, :]) for y in range(grid.shape[0])])

        return grid_str

    def __iter__(self):
        return iter((self.x, self.y))


class RoseMazeCMDP(FiniteCMDP):
    """
    Much like RoseMazeCPOMDP, except that the roses are definitely present
    The player must move around a 3x3m garden maze.
    The player (p) starts in the top left (0,0) and must move to the star (*) in the bottom left (2,2).
    Two squares (u,v) contain rose beds.
    A cost is incurred when the agent steps on a rose bed.

    # # # # #
    # p     #
    # @ @   #
    # *     #
    # # # # #

    States are represented by a tuple:
        (x, y) is the location of the agent

    There are 6 actions available:
        0: move right
        1: move down
        2: move left
        3: move up
    """

    S = FiniteSpace({
        XYState(x, y)
        for x in [0, 1, 2]
        for y in [0, 1, 2]
    })

    A = frozenset({IntAction(i) for i in [0, 1, 2, 3]})

    gamma = 0.9
    K = 1

    c_tuple = (0.314,)

    initial_state_dist = KroneckerDistribution(XYState(0, 0))

    def _inner_T(self, s: XYState, a: IntAction) -> Distribution:
        if self.is_sink(s):
            return KroneckerDistribution(s)
        else:
            if a not in self.A:
                raise ValueError
            x, y = s
            if a.n == 0 and x < 2:
                new_state = XYState(x + 1, y)
            elif a.n == 1 and y < 2:
                new_state = XYState(x, y + 1)
            elif a.n == 2 and x > 0:
                new_state = XYState(x - 1, y)
            elif a.n == 3 and y > 0:
                new_state = XYState(x, y - 1)
            else:
                new_state = s

            return KroneckerDistribution(new_state)

    def _inner_R(self, s, a) -> float:
        assert a in self.A
        if s == XYState(0, 1) and a.n == 1:
            return 1.0
        elif s == XYState(1, 2) and a.n == 2:
            return 1.0
        else:
            return 0.0

    def _inner_C(self, k: int, s: XYState, a: IntAction, ) -> float:
        x, y = s

        s_next_dist = self.T(s, a)
        # If T is deterministic, we know the next state
        if s_next_dist.is_degenerate():
            s_next = s_next_dist.sample()
        else:
            raise ValueError

        nx, ny = s_next
        assert k < self.K, f"k={k} is invalid, there are only K={self.K} cost functions"
        if (nx, ny) == (0, 1):
            return 1.0
        elif (nx, ny) == (1, 1):
            return 1.0
        else:
            return 0.0

    def is_sink(self, s) -> bool:
        x, y = s
        return (x, y) == (0, 2)
