import sys
import random
import os
from torch import manual_seed
from pather.main import pather
from gan_perceptron.main import gan_perceptron
from utilities import prepare_directory,find_enum_by_value
from drawer.main import draw_route
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
    ProgramModules.DRAWER:draw_route
}
module_enum = find_enum_by_value(ProgramModules,module)
os.environ['MODULE'] = module
prepare_directory(module_enum)
print(f"Running {module} module")
switch_dict[module_enum]()
# switch_dict[module_enum]('./output/finalActions/gan/generated_imgs/test.1000.0.txt')