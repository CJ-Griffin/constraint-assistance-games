# TODO (Maybe) ask Justin how to create a way to automatically validate the
#  inputs/outputs of T, R, C, c including:
#  * check for k<K
#  * check for s \in S, a \in A
#  * validate outputs
#  * if s is a sink then ...

# TODO create a CAG policy class, convert from BCMDP to CAG policy

# TODO maybe make special objects for States/Actions (belief states can inherit child and object)
#   (move rendering out of MDP)
#   consider making all States instances of dataclass (immutable)

# TODO consider writing a wrapping object X for numpy array A that:
#   * is callable via indeces as normal i.e. X[3, 1] = A[3, 1]
#   * but is also callable via states/actions i.e. X[s_3, s_1] = A[3, 1]
#   * it stores a fixed ordering of states and such to acheive this

# TODO consider speeding up cag_to_bcmdp by separating S and Beta
# TODO using lists instead of tuples for immutalbe lists (especially self.state_list etc)
