from src.formalisms.finite_processes import FiniteCAG
from src.reductions.cag_to_bcmdp import MatrixCAGtoBCMDP


def test_decision_process():
    import termios
    from src.concrete_processes import ALL_CONCRETE_DECISION_PROCESS_CLASSES
    from src.gym_env_wrapper import play_decision_process
    import enquiries

    try:
        DP = enquiries.choose('Choose a class to try: ', ALL_CONCRETE_DECISION_PROCESS_CLASSES)
        print(f"CHOSEN: {DP}")
    except termios.error as e:
        raise Exception("if you're running this in an IDE, try checking the box 'emulate terminal in output console'",
                        e)

    dp = DP()

    theta = None
    if isinstance(dp, FiniteCAG):
        should_reduce_to_BCMDP = enquiries.choose('Should reduce to BCMDP? ', [True, False])

        if should_reduce_to_BCMDP:
            dp = MatrixCAGtoBCMDP(dp)
        else:
            thetas = list(dp.Theta)
            theta = enquiries.choose('Choose from Θ: ', thetas)
            print(f"CHOSEN: {theta}")

    play_decision_process(dp, theta=theta)


if __name__ == "__main__":
    test_decision_process()
