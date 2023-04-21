import random

import numpy as np

from src.concrete_decision_processes.ecas_examples.dct_example import DCTRoseGardenCoop, DCTRoseGardenAppr, \
    ForbiddenFloraDCTCoop, ForbiddenFloraDCTAppr
from src.concrete_decision_processes.ecas_examples.pfd_example import SimplestFlowerFieldPFDCoop
from src.utils.policy_analysis import explore_mixed_CAG_policy_with_env_wrapper
from src.solution_methods.solvers import get_mixed_solution_to_FiniteCAG
from src.concrete_decision_processes.rose_garden_cags import RoseGarden, StochasticRoseGarden

example_cags = [
    RoseGarden(),  # The simplest CAG example
    StochasticRoseGarden(),  # A simple example requiring a mixed policy
    DCTRoseGardenAppr(),  # A simple apprenticeship example using ECAS
    DCTRoseGardenCoop(),  # A simple coop example using ECAS
    ForbiddenFloraDCTAppr(grid_size="extra_extra_large", size_of_Theta=2),  # A large example
    ForbiddenFloraDCTCoop(grid_size="medium", size_of_Theta=2),  # A medium size COOP example
    ForbiddenFloraDCTCoop(grid_size="small", size_of_Theta=3),  # A COOP example with |Θ|=3
    SimplestFlowerFieldPFDCoop()  # An example using PFD
]

for cag in example_cags:
    mixed_cag_policy = get_mixed_solution_to_FiniteCAG(cag)

    explore_mixed_CAG_policy_with_env_wrapper(
        mixed_cag_policy,
        cag,
        should_write_to_html=True
    )
