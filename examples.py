from src.concrete_decision_processes.ecas_examples.dct_example import DCTRoseGardenCoop, DCTRoseGardenAppr, \
    ForbiddenFloraDCTCoop
from src.concrete_decision_processes.ecas_examples.pfd_example import SimplestFlowerFieldPFDCoop
from src.utils.policy_analysis import explore_mixed_CAG_policy_with_env_wrapper
from src.solution_methods.solvers import get_mixed_solution_to_FiniteCAG
from src.concrete_decision_processes.rose_garden_cags import RoseGarden, StochasticRoseGarden

example_cags = [
    # RoseGarden(),
    # StochasticRoseGarden(),
    # DCTRoseGardenAppr(),
    # DCTRoseGardenCoop(),
    ForbiddenFloraDCTCoop(grid_size="medium", size_of_Theta=2),
    # SimplestFlowerFieldPFDCoop()
]

for cag in example_cags:
    mixed_cag_policy = get_mixed_solution_to_FiniteCAG(cag)

    explore_mixed_CAG_policy_with_env_wrapper(
        mixed_cag_policy,
        cag,
        should_write_to_html=True
    )
