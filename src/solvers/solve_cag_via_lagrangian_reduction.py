from src.solvers.lagrangian_cmdp_solver import naive_lagrangian_cmdp_solver
from src.formalisms import CAG, CAG_to_BMDP


def solve_cag_via_lagrangian_reduction(cag: CAG):
    bcmdp = CAG_to_BMDP(cag)
    naive_lagrangian_cmdp_solver(bcmdp)


