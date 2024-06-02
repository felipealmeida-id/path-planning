import argparse
import sys
import random
import os
from torch import manual_seed
# from pather.main import pather
from pather.maindos import pather
from gan_perceptron.main import gan_perceptron
from drawer.main import draw_route
from utilities import prepare_directory,find_enum_by_value
from enums import ProgramModules
from evaluator.main import evaluate_actions, evaluate_cartesian
# Import other necessary modules and functions

# Initialize the argparse
parser = argparse.ArgumentParser(description="Script Description")

# Adding arguments
parser.add_argument("-e", "--environment", required=True, help="Set the environment variable PY_ENV")
parser.add_argument("-m", "--module", required=True, help="Specify the module to run")
parser.add_argument("-p", "--path", help="Path, required for DRAWER or EVALUATOR modules", default='')
parser.add_argument("--profiler", action='store_true', help="Enable the profiler")
parser.add_argument("--mode", help="Set the mode for the evaluator, either 'actions' or 'cartesian'")
parser.add_argument("--res", help="Set the resolution, either HIGH or LOW")
# Parse the arguments
args = parser.parse_args()

# Set the environment variable for PY_ENV
os.environ["PY_ENV"] = args.environment

# Handle seed setting
random.seed(2023)
manual_seed(2023)

# Module processing
module_enum = find_enum_by_value(ProgramModules, args.module)
os.environ["MODULE"] = args.module
prepare_directory(module_enum)

if args.module in [ProgramModules.DRAWER.value, ProgramModules.EVALUATOR.value] and not args.path:
    parser.error("--path is required for DRAWER and EVALUATOR modules")

if args.module == ProgramModules.EVALUATOR.value and not args.mode:
    parser.error("--mode is required for EVALUATOR module")

if args.module in [ProgramModules.EVALUATOR.value, ProgramModules.DRAWER.value] and not args.res:
    parser.error("--res is required for EVALUATOR and DRAW module")



# Initialize specific environment based on the parsed arguments
# Here you would include your logic to choose the correct function to run based on the module_enum
# and whether to profile it or not.

switch_dict = {
    ProgramModules.PATHER: pather,
    ProgramModules.PERCEPTRON: gan_perceptron,
    ProgramModules.DRAWER: lambda: draw_route(args.path, args.res == 'HIGH'),
    ProgramModules.EVALUATOR: lambda: print(evaluate_actions(args.path, args.res)) if args.mode == 'actions' else print(evaluate_cartesian(args.path,args.res))
}

if args.profiler:
    # If --profiler is set, initialize and use your CustomProfiler on the chosen function
    from profiler.profiler import CustomProfiler
    profiler = CustomProfiler()
    profiler.profile_fun(switch_dict[module_enum])
else:
    # Run the chosen function without profiling
    switch_dict[module_enum]()