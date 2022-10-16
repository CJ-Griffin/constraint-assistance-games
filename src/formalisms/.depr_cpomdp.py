# from src.formalisms.distributions import *
# from abc import ABC, abstractmethod
#
#
# class CPOMDP(ABC):
#     S: set = None
#     A: set = None
#     Omega: set = None
#     gamma: float = None
#     K: int = None
#
#     b_0 : DiscreteDistribution = None
#
#     state = None
#
#     def perform_checks(self):
#         self.check_is_instantiated()
#         self.check_b0_is_valid()
#
#     @abstractmethod
#     def T(self, s, a) -> Distribution: # | None:
#         pass
#
#     @abstractmethod
#     def R(self, s, a) -> float:
#         pass
#
#     @abstractmethod
#     def O(self, a, s) -> Distribution:
#         pass
#
#     @abstractmethod
#     def C(self, k: int, s, a) -> float:
#         assert k < self.K, f"k={k} is invalid, there are only K={self.K} cost functions"
#         raise NotImplementedError
#
#     @abstractmethod
#     def c(self, k: int) -> float:
#         # this should be
#         # assert k < self.K, f"k={k} is invalid, there are only K={self.K} cost functions"
#         raise NotImplementedError
#
#     @abstractmethod
#     def is_sink(self, s) -> bool:
#         # this should be
#         # assert s in self.S, f"s={s} is not in S={self.S}"
#         raise NotImplementedError
#
#     def check_is_instantiated(self):
#         components = [
#             self.S,
#             self.A,
#             self.b_0,
#             self.Omega,
#             self.gamma,
#             self.K,
#         ]
#         if None in components:
#             raise ValueError("Something hasn't been instantiated!")
#
#     def check_b0_is_valid(self):
#         supp = set(self.b_0.support())
#         if not supp.issubset(self.S):
#             raise ValueError("b_0 should only have support over S!")
#
#     def render_state_as_string(self, s) -> str:
#         return str(s)
