try:
    # Initialize specific environment
    import sys
    from os import environ
    environ['PY_ENV'] = sys.argv[1]

    from random import seed
    from classes.environment import Environment
    from utils.env_parser import TOTAL_TIME
    
    seed(2023)
    env = Environment.get_instance()
    for i in range(TOTAL_TIME):
        env.iterate()
except ValueError as e:
    print(str(e.args[0]))
except KeyError as e:
    print(str(e.args[0]))