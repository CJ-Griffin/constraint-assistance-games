# DEPRECATED

# import numpy as np
#
# from src.formalisms.cpomdp import CPOMDP
# from src.formalisms.distributions import UniformDiscreteDistribution, Distribution, KroneckerDistribution
#
#
# class RoseMazeCPOMDP(CPOMDP):
#     """
#     The player must move around a 3x3m garden moze.
#     The player (p) starts in the top left (0,0) and must move to the star (*) in the bottom left (2,2).
#     Some squares (u,v) contain rose beds with probability 0.5.
#     A cost is incurred when the agent steps on a rose bed.
#
#     # # # # #
#     # p     #
#     # u v   #
#     # *     #
#     # # # # #
#
#     States are represented by a triple:
#         (u, v, x, y) where
#             u = 1 iff there is a rosebed at u
#             v = 1 iff there is a rosebed at u
#             loc is the coordinates of the agent
#
#     There are 6 actions available:
#         0: move right
#         1: move down
#         2: move left
#         3: move up
#         4: look down - gives perfect observation of whether there is a rose bed down
#
#     Observations contain the new state, but also an indicator of whether there is a rose bed ahead.
#         If the robot does not check, it will receive "NA"
#         If the robot checks and there are roses, it will receive "y"
#         If the robot checks and there are no roses, it will receieve "n"
#     """
#
#     S = {
#         (u, v, x, y)
#         for u in [0, 1]
#         for v in [0, 1]
#         for x in [0, 1, 2]
#         for y in [0, 1, 2]
#     }
#
#     A = {0, 1, 2, 3, 4}
#     Omega = {
#         (yn, x, y)
#         for x in [0, 1, 2]
#         for y in [0, 1, 2]
#         for yn in ["y", "n", "NA"]
#     }
#
#     gamma = 0.9
#     K = 1
#
#     b_0 = UniformDiscreteDistribution([
#         (1, 1, 0, 0)
#         # (u, v, 0, 0) for u in [0, 1] for v in [0, 1]
#     ])
#
#     def T(self, s, a) -> Distribution: # | None:
#         if a not in self.A:
#             raise ValueError
#         u, v, x, y = s
#         if a == 0 and x < 2:
#             new_state = (u, v, x + 1, y)
#         elif a == 1 and y < 2:
#             new_state = (u, v, x, y + 1)
#         elif a == 2 and x > 0:
#             new_state = (u, v, x - 1, y)
#         elif a == 3 and y > 0:
#             new_state = (u, v, x, y - 1)
#         else:
#             new_state = s
#
#         return KroneckerDistribution(new_state)
#
#     def R(self, s, a) -> float:
#         # Only works because it's deterministic!
#         next_s = self.T(s,a).sample()
#         if self.is_sink(next_s):
#             return 1.0
#         else:
#             return 0.0
#
#     def O(self, a, sp) -> Distribution:
#         u, v, x, y = sp
#         if a == 4 and x == 0 and y == 0:
#             o0 = "y" if u == 1 else "n"
#         elif a == 4 and x == 1 and y == 0:
#             o0 = "y" if v == 1 else "n"
#         else:
#             o0 = "NA"
#         o = o0, x, y
#         assert o in self.Omega
#         return KroneckerDistribution(o)
#
#     def C(self, k: int, s, a) -> float:
#         u, v, x, y = s
#         # Only works because it's deterministic!
#         next_s = self.T(s,a).sample()
#         _, _, nx, ny = next_s
#         assert k < self.K, f"k={k} is invalid, there are only K={self.K} cost functions"
#         if u == 1 and (nx, ny) == (0, 1):
#             return 1.0
#         elif v == 1 and (nx, ny) == (1, 1):
#             return 1.0
#         else:
#             return 0.0
#
#     def c(self, k: int) -> float:
#         return 0.5
#
#     def is_sink(self, s) -> bool:
#         u, v, x, y = s
#         return (x, y) == (0, 2)
#
#     def render_state_as_string(self, s) -> str:
#         u, v, x, y = s
#         grid = np.array([
#             [".", ".", "."],
#             ["@" if u == 1 else ".", "@" if v == 1 else ".", "."],
#             ["*", ".", "."],
#         ])
#         grid[y,x] = "p"
#
#         grid_str = "\n".join([" ".join(grid[y, :]) for y in range(grid.shape[0])])
#
#         return grid_str
