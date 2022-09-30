"""
Deprecated, since it's easier to do CAG -> BCMDP directly
"""
# import numpy as np
#
# from src.formalisms.cmdp import CMDP
# from src.formalisms.cpomdp import CPOMDP
# from src.formalisms.distributions import Distribution, KroneckerDistribution, DiscreteDistribution
# from src.formalisms.spaces import AllDistributionsOverFiniteSet
#
#
# class NoPossibleBeliefState(Exception):
#     def __init__(self, obs, action):
#         self.obs = obs
#         self.action = action
#         message = f"observation {obs} cannot result from action {action}"
#         super().__init__(message)
#
#
# class BCMDP(CMDP):
#
#     def __init__(self, cpomdp: CPOMDP):
#         self.cpomdp = cpomdp
#         self.belief_space = AllDistributionsOverFiniteSet(cpomdp.S)
#         self.S = self.belief_space
#         self.A = cpomdp.A
#         self.gamma = cpomdp.gamma
#         self.K = cpomdp.K
#
#         self.I = KroneckerDistribution(cpomdp.b_0)
#
#         self.perform_checks()
#
#         self.NoBeliefState =
#
#     def T(self, b, a) -> Distribution:  # | None:
#         prob_bp = {}
#         for o in self.cpomdp.Omega:
#             next_b_given_o, prob_o_given_a_b = self._belief_update_and_denominator(b, a, o)
#             # TODO when you pick this up
#             #  - implement equality between distributions
#             #  - finish this function (it might be done)
#
#             if next_b_given_o in prob_bp:
#                 prob_bp[next_b_given_o] += prob_o_given_a_b
#             else:
#                 prob_bp[next_b_given_o] = prob_o_given_a_b
#
#         return DiscreteDistribution(prob_bp)
#
#     def R(self, b, a, next_b) -> float:
#         concrete_states = list(self.cpomdp.S)
#
#         poss_obs = [
#             o for o in self.cpomdp.Omega
#             if self._belief_update_and_denominator(b, a, o)[0] == next_b
#         ]
#
#         poss_triplets = [
#             (s, o, next_s)
#             for s in concrete_states
#             for o in poss_obs
#             for next_s in concrete_states
#         ]
#
#         # Function assumes o would update us from b to bp
#         def prob_s_o_sp_given_b_a_bp(s, o, sp):
#             term1 = self.cpomdp.O(a, sp).get_probability(o)
#             term2 = self.cpomdp.T(s, a).get_probability(sp)
#             term3 = b.get_probability(s)
#             return term1 * term2 * term3
#
#         vals = [prob_s_o_sp_given_b_a_bp(s, o, sp) for (s, o, sp) in poss_triplets]
#         total = sum(vals)
#
#     def C(self, k: int, b, a, next_b) -> float:
#         pass
#
#     def c(self, k: int) -> float:
#         return self.cpomdp.c(k)
#
#     def is_sink(self, b) -> bool:
#         pass
#
#     def _belief_update_and_denominator(self,
#                                        b: Distribution,
#                                        a,
#                                        o
#                                        ) -> (Distribution, float):
#         """
#         Note: if P[a, o | s] = 0 for all s s.t. b(s) > 0, then b'[a,o] is undef.
#         :param b:
#         :param a:
#         :param o:
#         :return:
#         """
#         assert b in self.belief_space
#         assert a in self.A
#         assert o in self.cpomdp.Omega
#
#         # P[ s' | o, a, b] = P[ o | a, s'] * P[s' | a, b] / P[o | a, b]
#         # i.e.             = ------X------ * -----Y------ / ------Z-----
#         concrete_states = list(self.cpomdp.S)
#
#         Xs = [
#             self.cpomdp.O(a, s_next).get_probability(o)
#             for s_next in concrete_states
#         ]
#         Ys = [
#             sum([
#                 self.cpomdp.T(s, a).get_probability(s_next) * b.get_probability(s)
#                 for s in concrete_states
#             ])
#             for s_next in concrete_states
#         ]
#         numerators = [Xs[i] * Ys[i] for i in range(len(Xs))]
#         Z = sum(numerators)
#
#         if Z == 0:
#             if sum(Xs) == 0:
#                 x_map = {
#                     concrete_states[i]: Xs[i]
#                     for i in range(len(concrete_states))
#                 }
#                 y_map = {
#                     concrete_states[i]: Ys[i]
#                     for i in range(len(concrete_states))
#                 }
#                 # raise ValueError(f"Since action O(o | a, s') = P[o|a, s'] = 0 for all s', belief state is undefined")
#                 raise NoPossibleBeliefState(obs=o, action=a)
#             else:
#                 raise NotImplementedError()
#         b_next_map = {
#             concrete_states[i]: numerators[i] / Z
#             for i in range(len(concrete_states))
#         }
#         return DiscreteDistribution(b_next_map), Z
#
#         # # i.e. next_b_sp = X * Y / Z
#         # # Z is a constant normalising factor, which can be calculated at the end
#         # numerators = {}
#         #
#         # concrete_states = list(self.cpomdp.S)
#         #
#         # t_dists = {
#         #     s: self.cpomdp.T(s, a)
#         #     for s in concrete_states
#         # }
#         # t_ps = {
#         #     (s, s_next): t_dists[s].get_probability(s_next)
#         #     for s in concrete_states
#         #     for s_next in concrete_states
#         #
#         # }
#         #
#         # # t_ps_matrix[index of s, index of s_next] == T( s_next | s, a)
#         # t_ps_matrix = np.array(
#         #     [[t_ps[(s, s_next)] for s_next in concrete_states] for s in concrete_states]
#         # )
#         # row_sum = t_ps_matrix.sum(axis=1)
#         #
#         # # For each s, \Sum_{s_next}( T(s_next | s, a) ) = 1
#         # is_row_stochastic = (row_sum == 1.0).all()
#         #
#         # assert is_row_stochastic, f"{row_sum}"
#         #
#         # # b_vec[index_of_s_next] = P [ s_next | s, a ]
#         # b_vec = np.array([b.get_probability(s) for s in concrete_states])
#         #
#         # # b_next(s_next) = P [ s_next | s, a ] * P [o
#         #
#         # assert b_vec.sum() == 1.0
#         #
#         # b_next_vec = t_ps_matrix.transpose() @ b_vec
#         #
#         # # for s_next in self.cpomdp.S:
#         # #     X = self.cpomdp.O(a, s_next).get_probability(o)
#         # #
#         # #     ts = [self.cpomdp.T(s, a).get_probability(s_next)
#         # #           for s in self.cpomdp.S]
#         # #     bs = [b.get_probability(s)
#         # #           for s in self.cpomdp.S]
#         # #     y_terms = [bs[i] * ts[i] for i in range(len(ts))]
#         # #     Y = sum(y_terms)
#         # #     numerators[s_next] = X * Y
#         # #
#         # # Z = sum(list(numerators.values()))
#         # # next_b_map = {
#         # #     numerators[s_next] / Z
#         # #     for s_next in self.cpomdp.S
#         # # }
#         # return DiscreteDistribution(next_b_map), Z
