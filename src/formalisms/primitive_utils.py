from src.formalisms.distributions import Distribution, FiniteParameterDistribution, DiscreteDistribution


def split_initial_dist_into_s_and_beta(joint_initial_dist: Distribution) -> (object, Distribution):
    if isinstance(joint_initial_dist, FiniteParameterDistribution):
        raise NotImplementedError
    elif isinstance(joint_initial_dist, DiscreteDistribution):
        sup = joint_initial_dist.support()
        support_over_states = {
            s for (s, theta) in sup
        }
        if len(support_over_states) != 1:
            raise ValueError(f"Reduction to coordination BCMDP only supported when s_0 is deterministic:"
                             f" dist.support()={sup}")
        else:
            s = list(support_over_states)[0]

        theta_map = {
            theta: joint_initial_dist.get_probability((s, theta))
            for _, theta in joint_initial_dist.support()
        }

        b = DiscreteDistribution(theta_map)
        beta = FiniteParameterDistribution(b, frozenset(b.support()))

        return s, beta
    else:
        raise NotImplementedError
