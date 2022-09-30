from abc import abstractmethod, ABC
from collections.abc import Iterable
from typing import Sized

import numpy as np

from src.formalisms.distributions import DiscreteDistribution
from itertools import chain, combinations


class Space(ABC):
    """

    """

    # def copy(self, *args, **kwargs):  # real signature unknown
    #     """ Return a shallow copy of a set. """
    #     pass

    # def difference(self, *args, **kwargs): # real signature unknown
    #     """
    #     Return the difference of two or more sets as a new set.
    #
    #     (i.e. all elements that are in this set but not the others.)
    #     """
    #     pass

    # def intersection(self, other_set):  # real signature unknown
    #     """
    #     Return the intersection of two sets as a new set.
    #
    #     (i.e. all elements that are in both sets.)
    #     """
    #     pass

    #
    # def isdisjoint(self, *args, **kwargs): # real signature unknown
    #     """ Return True if two sets have a null intersection. """
    #     pass
    #
    # def issubset(self, *args, **kwargs): # real signature unknown
    #     """ Report whether another set contains this set. """
    #     pass
    #
    # def issuperset(self, *args, **kwargs): # real signature unknown
    #     """ Report whether this set contains another set. """
    #     pass
    #
    # def symmetric_difference(self, *args, **kwargs):  # real signature unknown
    #     """
    #     Return the symmetric difference of two sets as a new set.
    #
    #     (i.e. all elements that are in exactly one of the sets.)
    #     """
    #     pass
    #
    # def union(self, otherset):  # real signature unknown
    #     """
    #     Return the union of sets as a new set.
    #
    #     (i.e. all elements that are in either set.)
    #     """
    #     return UnionSpace(self, otherset)
    #
    # def __and__(self, *args, **kwargs):  # real signature unknown
    #     """ Return self&value. """
    #     pass
    #
    # def __class_getitem__(self, *args, **kwargs):  # real signature unknown
    #     """ See PEP 585 """
    #     pass

    @abstractmethod
    def __contains__(self, y):  # real signature unknown; restored from __doc__
        """ x.__contains__(y) <==> y in x. """
        pass

    # def __eq__(self, *args, **kwargs): # real signature unknown
    #     """ Return self==value. """
    #     pass
    #
    # def __hash__(self, *args, **kwargs):  # real signature unknown
    #     """ Return hash(self). """
    #     pass

    # def __len__(self, *args, **kwargs): # real signature unknown
    #     """ Return len(self). """
    #     pass


class UnionSpace(Space):
    def __init__(self, space1, space2):
        self.space1 = space1
        self.space2 = space2

    def __contains__(self, item):
        return item in self.space1 or item in self.space2


class FiniteSpace(Space, Iterable, Sized):
    is_finite = True

    def __init__(self, set_of_elements):
        super().__init__()
        self.set = set_of_elements

    def __contains__(self, item):
        return item in self.set

    def __len__(self):
        return len(self.set)

    def __iter__(self):
        return iter(self.set)




# class ReachableStatesAndBetas(FiniteSpace):
#     def __init__(self, S: set, beta_0: DiscreteDistribution):
#         Theta = beta_0.support()
#         self.prior = beta_0
#         set_of_betas = {
#             DiscreteDistribution(self._get_distr_from_subset(sset=sset))
#             for sset in powerset(Theta)
#         }
#         set_of_pairs = {
#             (s, beta)
#         }
#         super().__init__(set_of_betas)
#
#     def _get_distr_from_subset(self, sset: set, bete):
#         elems = list(sset)
#         priors = np.array([self.prior.get_probability(e) for e in sset])
#         probs = priors / priors.sum()
#
#         return DiscreteDistribution({
#             elems[i]: probs[i] for i in range(len(elems))
#         })


class InfiniteSpace(Space):
    is_finite = False

    def __contains__(self, y):
        pass

    def __len__(self):
        raise ValueError


class AllDistributionsOverFiniteSet(Space):
    def __init__(self, fin_set: set):
        self.fin_set = fin_set

    def __contains__(self, item):
        # Reject if the item is not a distribution over a finite set
        if not isinstance(item, DiscreteDistribution):
            return False
        # Reject if the item has support beyond self.fin_set
        elif set(item.support()).issubset(self.fin_set):
            return True
        # Otherwise the item is a distribution with
        # support over (at most) self.fin_set
        else:
            return False
