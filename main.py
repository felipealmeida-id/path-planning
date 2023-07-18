try:
    # Initialize specific environment
    import sys
    from os import environ
    environ['PY_ENV'] = sys.argv[1]

    from random import seed
    from classes.environment import Environment
    
    seed(2023)
    env = Environment.get_instance()
    env.uavs[0].possible_moves()
    env.iterate()
    env.iterate()
    env.iterate()
    env.iterate()
except ValueError as e:
    print(str(e.args[0]))
except KeyError as e:
    print(str(e.args[0]))