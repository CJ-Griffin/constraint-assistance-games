from src.appr_grid_cag import ApprenticeshipStaticGridCAG, ASGState
from src.formalisms.distributions import UniformDiscreteDistribution


class RoseGarden(ApprenticeshipStaticGridCAG):
    K = 1
    Theta = {"imprm", "prm"}

    def __init__(self):
        super().__init__(
            h_height=3,
            h_width=2,
            h_start=(0, 2),
            h_sinks={(0, 0)},
            r_height=2,
            r_width=3,
            r_start=(0, 1),
            r_sinks={(2, 1)},
            goal_reward=1.0,
            gamma=0.9)
        self.I = UniformDiscreteDistribution({(self.s_0, theta) for theta in self.Theta})
        self.perform_checks()

    def c(self, k: int) -> float:
        assert k < self.K, f"k={k} is invalid, there are only K={self.K} cost functions"
        return 0.5

    def C(self, k: int, theta, s: ASGState, h_a, r_a) -> float:
        assert k < self.K, f"k={k} is invalid, there are only K={self.K} cost functions"
        assert theta in self.Theta, f"theta={theta} is invalid, it's not in Theta={self.Theta}"

        # NOTE: this only works because this CAG is deterministic!
        next_s = self.T(s, h_a, r_a).sample()
        if theta == "prm":
            return 0.0
        else:
            assert theta == "imprm"
            if next_s.h_xy == (0, 1):
                return 1.0
            elif next_s.r_xy == (1, 1):
                return 1.0
            else:
                return 0.0

    def render_state_as_string(self, s: ASGState) -> str:
        (h_x, h_y) = s.h_xy
        (r_x, r_y) = s.r_xy
        whose_turn = s.whose_turn
        import numpy as np
        hum_grid = np.array([
            ["*", "."],
            ["@", "."],
            [".", "."],
        ])

        hum_grid[h_y, h_x] = "h"

        hum_grid = np.hstack([
            hum_grid,
            np.array([["#"], ["#"], ["#"]]),
            np.array([["|"], ["|"], ["|"]]),
            np.array([["#"], ["#"], ["#"]]),
        ])

        rob_grid = np.array([
            [".", ".", "."],
            [".", "@", "*"],
        ])
        rob_grid[r_y, r_x] = "r"

        rob_grid = np.vstack([
            rob_grid,
            np.array(["#", "#", "#"])
        ])

        comb = np.hstack([hum_grid, rob_grid])
        top_bottom = np.array(["#"] * comb.shape[1])
        comb = np.vstack([
            top_bottom,
            comb,
            top_bottom
        ])
        left = np.array([["#"]] * comb.shape[0])
        right = np.array([["#"]] * comb.shape[0])

        comb = np.hstack([
            left,
            comb,
            right
        ])

        comb_str = "\n".join([" ".join(comb[y, :]) for y in range(comb.shape[0])])
        comb_str += f"\n {whose_turn}'s turn"
        return comb_str
