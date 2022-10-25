import unittest
from abc import ABC, abstractmethod

from src.example_environments.a_cmdp_that_requires_a_stochastic_policy import ACMDPNeedingStochasticity, \
    ASecondCMDPNeedingStochasticity
from src.example_environments.maze_cmdp import RoseMazeCMDP
from src.example_environments.randomised_cags_and_cmdps import RandomisedCMDP, RandomisedCAG
from src.example_environments.rose_garden_cag import RoseGarden
from src.example_environments.simple_mdp import SimpleMDP
from src.example_environments.simplest_cag import SimplestCAG
from src.formalisms.decision_process import DecisionProcess


class TestProcess(ABC):

    def setUp(self) -> None:
        pass

    @abstractmethod
    def create_process(self) -> DecisionProcess:
        pass

    def test_process(self):
        process = self.create_process()
        process.enable_debug_mode()


class TestRoseGarden(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return RoseGarden()


class TestRoseMazeCMDP(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return RoseMazeCMDP()


class TestSimpleMDP(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return SimpleMDP()


class TestRandCMDP(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return RandomisedCMDP(K=4,
                              max_steps=3,
                              max_x=4,
                              num_a=2)


class TestRandCAG(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return RandomisedCAG(K=4,
                             max_steps=3,
                             max_x=4)


class TestSimplestCAG(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return SimplestCAG()


class TestStochCMDP1(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return ACMDPNeedingStochasticity()


class TestStochCMDP2(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return ASecondCMDPNeedingStochasticity()
