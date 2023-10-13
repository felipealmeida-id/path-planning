import sys
import random
import os
from torch import manual_seed
from pather.main import pather
from gan_perceptron.main import gan_perceptron
from cartesianGAN.main import gan_cartesian
from drawer.main import draw_route
from utilities import prepare_directory,find_enum_by_value
from enums import ProgramModules

# Initialize specific environment
os.environ["PY_ENV"] = sys.argv[1]
random.seed(2023)
manual_seed(2023)

module = sys.argv[2]
module_enum = find_enum_by_value(ProgramModules, module)
os.environ["MODULE"] = module
draw_path = sys.argv[3] if module_enum == ProgramModules.DRAWER else ''
prepare_directory()
switch_dict = {
    ProgramModules.PATHER:pather,
    ProgramModules.PERCEPTRON:gan_perceptron,
    ProgramModules.CARTESIAN:gan_cartesian,
    ProgramModules.DRAWER:lambda: draw_route(draw_path),
    ProgramModules.EVALUATOR:""
}
if sys.argv[-1] == 'p':
    from profiler.profiler import CustomProfiler
    profiler = CustomProfiler()
    profiler.profile_fun(switch_dict[module_enum])
else:
    switch_dict[module_enum]()
