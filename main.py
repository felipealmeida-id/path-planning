import sys
import random
import os
from torch import manual_seed
from pather.main import pather
from gan_perceptron.main import gan_perceptron
from env_parser import Env
# Initialize specific environment
os.environ['PY_ENV'] = sys.argv[1]
random.seed(2023)
manual_seed(2023)

module = sys.argv[2]

switch_dict = {
    "pather":pather,
    "gan":gan_perceptron
}

if module in switch_dict:
    os.environ['MODULE'] = module
    env = Env.get_instance()
    switch_dict[module]()
else:
    print(f"Invalid command \"{module}\"")
    print(f'Valid commands are: {list(switch_dict.keys())}')