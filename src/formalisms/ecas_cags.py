from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import FrozenSet, Hashable, Set, Callable

from src.formalisms.cag import CAG, FiniteCAG
from src.formalisms.decision_process import validate_c
from src.formalisms.distributions import DiscreteDistribution


class EthicalContext(Hashable, ABC):
    nickname: str

    # Both of these methods will be redefined by dataclass
    def __hash__(self) -> int:
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def render(self, short: bool = True):
        if short:
            return f"{self.__class__.__name__}< {self.nickname}>"
        else:
            rend_str = f"{self.__class__.__name__}("
            for field in fields(self):
                rend_str += f"{field.name}:{getattr(self, field.name)}, "
            rend_str += ")"
            return rend_str


class EthicalComplianceAG(CAG, ABC):
    Theta: Set[EthicalContext]

    @abstractmethod
    def C(self, k: int, theta: EthicalContext, s, h_a, r_a) -> float:
        raise NotImplementedError


@dataclass(frozen=True, eq=True)
class DCTEthicalContext(EthicalContext):
    forbidden_states: FrozenSet
    nickname: str


class DivineCommandTheoryCAG(EthicalComplianceAG, ABC):
    K = 1

    def C(self, k: int, theta: DCTEthicalContext, s, h_a, r_a) -> float:
        if theta not in self.Theta:
            raise ValueError
        elif k != 0:
            raise ValueError
        else:
            next_state_dist = self.T(s, (h_a, r_a))
            forbidden = {f for f in next_state_dist.support() if f in theta.forbidden_states}
            return sum(next_state_dist.get_probability(f) for f in forbidden)

    @validate_c
    def c(self, k: int) -> float:
        return 0.0

    def __init__(self, ethical_contexts: FrozenSet[DCTEthicalContext]):
        self.Theta: Set[DCTEthicalContext] = set(ethical_contexts)
        for ec in self.Theta:
            for f in ec.forbidden_states:
                if f not in self.S:
                    raise ValueError


@dataclass(frozen=True, eq=True)
class PFDEthicalContext(EthicalContext):
    # Δ = set of duties
    duties: FrozenSet[str]

    # φ is the penalty function
    # φ : S x Δ -> R
    penalty_function: Callable[[object, str], float]

    # τ is a tolerance
    tolerance: float

    nickname: str

    def __post_init__(self):
        if self.tolerance < 0:
            raise ValueError


# We have to restrict PFD to FiniteCAG so that we can take an expectation over the next state in C(k, θ, s, ah, ar)
class PrimaFacieDutiesCAG(EthicalComplianceAG, FiniteCAG, ABC):
    K: int = 1

    def __init__(self, ethical_contexts: FrozenSet[PFDEthicalContext]):
        self.Theta: FrozenSet[PFDEthicalContext] = ethical_contexts

    def C(self, k: int, theta: PFDEthicalContext, s, h_a, r_a) -> float:
        if k != 0:
            raise ValueError

        # Get a distribution over s'
        t_next_dist: DiscreteDistribution = self.T(s, (h_a, r_a))

        # f(s') = Σ_{δ ∈ Δ} φ(s', δ)

        deltas = list(theta.duties)

        def aggregate_penalty_function(s_next):
            penalties = [theta.penalty_function(s_next, delta) for delta in deltas]
            return sum(penalties)

        # Exp_{s'} [ f(s') ] = Exp_{s'} [ Σ_{δ ∈ Δ} φ(s', δ) ]
        expected_aggregate_penalty = t_next_dist.expectation(aggregate_penalty_function)

        # Normalise relative to the tolerance
        normalised_penalty = expected_aggregate_penalty / theta.tolerance

        return normalised_penalty

    @validate_c
    def c(self, k: int) -> float:
        return 1.0
