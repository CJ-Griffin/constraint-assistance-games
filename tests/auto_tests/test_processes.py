import unittest
from abc import ABC, abstractmethod

from src.concrete_processes.a_cmdp_that_requires_a_stochastic_policy import ACMDPNeedingStochasticity, \
    ASecondCMDPNeedingStochasticity
from src.concrete_processes.ecas_examples.dct_example import ForbiddenFloraDCTCoop, SmallForbiddenFloraDCT, \
    MediumForbiddenFloraDCT
from src.concrete_processes.ecas_examples.pfd_example import FlowerFieldPFDCoop, MediumFlowerFieldPFDCoop, \
    SmallFlowerFieldPFDCoop
from src.concrete_processes.maze_cmdp import RoseMazeCMDP
from src.concrete_processes.randomised_cags_and_cmdps import RandomisedCMDP, RandomisedCAG
from src.concrete_processes.rose_garden_cags import RoseGarden, CoopRoseGarden, SimplestCAG
from src.concrete_processes.simple_mdp import SimpleMDP
from src.formalisms.abstract_decision_processes import DecisionProcess, CAG, CMDP
from src.formalisms.policy import RandomCAGPolicy, RandomCMDPPolicy
from src.policy_analysis import explore_CAG_policy_with_env_wrapper, explore_CMDP_policy_with_env_wrapper


class TestProcess(ABC):

    def setUp(self) -> None:
        pass

    @abstractmethod
    def create_process(self) -> DecisionProcess:
        pass

    def test_process(self):
        process = self.create_process()
        process.perform_checks()

    def test_against_random(self):
        process = self.create_process()

        # If the CAG is too large, skip this test
        if len(process.S) > 50:
            return

        if isinstance(process, CAG):
            cag = process
            policy = RandomCAGPolicy(S=cag.S, h_A=cag.h_A, r_A=cag.r_A)
            explore_CAG_policy_with_env_wrapper(policy, cag=cag, should_render=False, max_runs=1)
        elif isinstance(process, CMDP):
            policy = RandomCMDPPolicy(process.S, process.A)
            explore_CMDP_policy_with_env_wrapper(policy=policy, cmdp=process, should_render=False)


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


class TestForbiddenFlora(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return ForbiddenFloraDCTCoop()


class TestCooperative(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return CoopRoseGarden()


class TestDCT(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return ForbiddenFloraDCTCoop(grid_size="tiny")


class TestDCTSmall(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return SmallForbiddenFloraDCT()


class TestDCTMedium(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return MediumForbiddenFloraDCT()


class TestPFD(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return FlowerFieldPFDCoop()


class TestPFDSmall(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return SmallFlowerFieldPFDCoop()


class TestPFDMedium(TestProcess, unittest.TestCase):
    def create_process(self) -> DecisionProcess:
        return MediumFlowerFieldPFDCoop()
