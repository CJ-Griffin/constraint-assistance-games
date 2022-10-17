# TODO (Maybe) ask Justin how to create a way to automatically validate the
#  inputs/outputs of T, R, C, c including:
#  * check for k<K
#  * check for s \in S, a \in A
#  * validate outputs
#  * if s is a sink then ...

# TODO make a policy class (update in get_traj_dist, etc)
# TODO convert CMDP policy to CAG policy

# TODO add test for get_traj_dist

# TODO maybe make special objects for States/Actions (belief states can inherit child and object)
#   (move rendering out of MDP)
#   consider making all States instances of dataclass (immutable)
