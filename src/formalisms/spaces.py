from abc import abstractmethod, ABC
from collections.abc import Iterable
from typing import Sized


class Space(ABC, Iterable):
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

    # def __len__(self, *args, **kwargs) -> float:
    #     """
    #     Return len(self).
    #     If the space is infinite, return float('inf')
    #
    #     """
    #     pass


# class UnionSpace(Space):
#     def __init__(self, space1, space2):
#         self.space1 = space1
#         self.space2 = space2
#
#     def __contains__(self, item):
#         return item in self.space1 or item in self.space2
#
#     def __iter__(self):
#         return zip(iter(self.space1), iter(self.space2))


class FiniteSpace(Space, Iterable, Sized):
    is_finite = True

    def __init__(self, set_of_elements):
        super().__init__()
        self._set = set_of_elements
        self._len = len(set_of_elements)

    def __contains__(self, item):
        return item in self._set

    def __len__(self):
        return self._len

    def __iter__(self):
        return iter(self._set)


class CountableSpace(Space, Iterable):

    @abstractmethod
    def __contains__(self, y):  # real signature unknown; restored from __doc__
        """ x.__contains__(y) <==> y in x. """
        pass

    @abstractmethod
    def __iter__(self):
        pass
