import numpy as np

from src.formalisms.spaces import FiniteSpace
from src.formalisms.cmdp import CMDP, FiniteCMDP
from src.formalisms.distributions import Distribution, KroneckerDistribution, UniformDiscreteDistribution


class RoseMazeCMDP(FiniteCMDP):
    """
    Much like RoseMazeCPOMDP, except that the roses are definitely present
    The player must move around a 3x3m garden moze.
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
        (x, y)
        for x in [0, 1, 2]
        for y in [0, 1, 2]
    })

    A = {0, 1, 2, 3}

    gamma = 0.9
    K = 1

    I = KroneckerDistribution((0, 0))

    def T(self, s, a) -> Distribution: # | None:
        if a not in self.A:
            raise ValueError
        x, y = s
        if a == 0 and x < 2:
            new_state = (x + 1, y)
        elif a == 1 and y < 2:
            new_state = (x, y + 1)
        elif a == 2 and x > 0:
            new_state = (x - 1, y)
        elif a == 3 and y > 0:
            new_state = (x, y - 1)
        else:
            new_state = s

        return KroneckerDistribution(new_state)

    def R(self, s, a) -> float:
        assert a in self.A
        if s == (0,1) and a == 1:
            return 1.0
        elif s == (1,2) and a == 2:
            return 1.0
        else:
            return 0.0

    def C(self, k: int, s, a,) -> float:
        x, y = s
        # Only works because it's deterministic!
        next_s = self.T(s,a).sample()
        nx, ny = next_s
        assert k < self.K, f"k={k} is invalid, there are only K={self.K} cost functions"
        if (nx, ny) == (0, 1):
            return 1.0
        elif (nx, ny) == (1, 1):
            return 1.0
        else:
            return 0.0

    def c(self, k: int) -> float:
        return 0.5

    def is_sink(self, s) -> bool:
        x, y = s
        return (x, y) == (0, 2)

    def render_state_as_string(self, s) -> str:
        x, y = s
        grid = np.array([
            [".", ".", "."],
            ["@", "@", "."],
            ["*", ".", "."],
        ])
        grid[y, x] = "p"

        grid_str = "\n".join([" ".join(grid[y, :]) for y in range(grid.shape[0])])

        return grid_str
