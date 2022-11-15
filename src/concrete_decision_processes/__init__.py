def _get_all_decision_proccesses():
    import inspect
    import os
    import pkgutil

    from src.formalisms.abstract_decision_processes import DecisionProcess

    env_path = os.path.split(os.path.abspath(__file__))[0]

    _dp_classes = []

    __all__ = []
    for loader, module_name, is_pkg in pkgutil.walk_packages([env_path]):
        __all__.append(module_name)
        _module = loader.find_module(module_name).load_module(module_name)
        for name in dir(_module):
            item = getattr(_module, name)
            if inspect.isclass(item) and issubclass(item, DecisionProcess):
                if not inspect.isabstract(item):
                    _dp_classes.append(item)
    return _dp_classes


ALL_CONCRETE_DECISION_PROCESS_CLASSES = _get_all_decision_proccesses()
