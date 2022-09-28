from src.formalisms import DiscreteDistribution

if __name__ == "__main__":
    dist1 = DiscreteDistribution({"a": 0.2, "b": 0.3, "c": 0.5})
    dist2 = DiscreteDistribution({"a": 0.2, "b": 0.3, "c": 0.5, "d": 0.0})
    try:
        dist3 = DiscreteDistribution({"a": 0.2, "b": 0.3, "c": 0.5, "d": 0.1})
    except Exception as e:
        print(e)
    try:
        dist4 = DiscreteDistribution({"a": 0.1, "b": 0.3, "c": 0.5, "d": 0.0})
    except Exception as e:
        print(e)

    assert dist1 == dist2, f"{dist1} != {dist2}"
