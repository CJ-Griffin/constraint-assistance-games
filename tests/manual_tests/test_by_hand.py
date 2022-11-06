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
    play_decision_process(dp)


if __name__ == "__main__":
    test_decision_process()
