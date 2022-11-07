from src.formalisms.abstract_decision_processes import CAG


def test_decision_process():
    import termios
    from src.concrete_processes import ALL_CONCRETE_DECISION_PROCESS_CLASSES
    from src.env_wrapper import play_decision_process
    import enquiries

    try:
        DP = enquiries.choose('Choose a class to try: ', ALL_CONCRETE_DECISION_PROCESS_CLASSES)
        print(f"CHOSEN: {DP}")
    except termios.error as e:
        raise Exception("if you're running this in an IDE, try checking the box 'emulate terminal in output console'",
                        e)

    dp = DP()

    theta = None
    if isinstance(dp, CAG):
        try:
            thetas = list(dp.Theta)
            theta = enquiries.choose('Choose from Θ: ', thetas)
            print(f"CHOSEN: {theta}")
        except termios.error as e:
            raise Exception(
                "if you're running this in an IDE, try checking the box 'emulate terminal in output console'",
                e)

    play_decision_process(dp, theta=theta)


if __name__ == "__main__":
    test_decision_process()
