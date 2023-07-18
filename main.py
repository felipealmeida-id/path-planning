from random import seed
from classes.environment import Environment

seed(2023)
env = Environment.get_instance()
env.uavs[0].possible_moves()
env.iterate()
env.iterate()
env.iterate()
env.iterate()