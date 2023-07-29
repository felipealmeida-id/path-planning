import sys
import random
import os
from torch import manual_seed
from pather.main import pather
from gan_perceptron.main import gan_perceptron
from utilities import prepare_directory
from enums import ProgramModules
# Initialize specific environment
os.environ['PY_ENV'] = sys.argv[1]
random.seed(2023)
manual_seed(2023)

module = sys.argv[2]

switch_dict = {
    ProgramModules.PATHER:pather,
    ProgramModules.PERCEPTRON:gan_perceptron,
    ProgramModules.EVALUATOR:"",
    ProgramModules.DRAWER:""
}

if module in switch_dict:
    os.environ['MODULE'] = module
    prepare_directory()
    switch_dict[module]()
else:
    print(f"Invalid command \"{module}\"")
    print(f'Valid commands are: {list(switch_dict.keys())}')