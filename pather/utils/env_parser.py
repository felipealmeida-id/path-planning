from dotenv import load_dotenv
from typing import Dict,Callable
from json import loads
from os import getenv
from os.path import exists as file_exists

def non_negative_integer(args:tuple[str,bool]):
    x:str = args[0]
    success:bool = args[1] if len(args) > 1 else True
    return x,(success and x.isdigit())

def non_positive_integer(args:tuple[str,bool]):
    x:str = args[0]
    success:bool = args[1] if len(args) > 1 else True
    return x,(success and x[0]=='-' and x[1:].isdigit())

def non_zero(args:tuple[str,bool]):
    x:str = args[0]
    success:bool = args[1] if len(args) > 1 else True
    return x,(success and x!='0')

def greater_than_zero(args:tuple[str,bool]):
    x = args[0]
    success = args[1] if len(args) > 1 else True
    return non_zero(non_negative_integer((x,success)))

def coord_list(args:tuple[str,bool]):
    x = args[0]
    success = args[1] if len(args) > 1 else True
    try:
        data = loads(x)
        if not isinstance(data,list):
            return x,False
        for coord in data:
            if not (isinstance(coord["x"],int) and isinstance(coord["y"],int)):
                return x,False
    except:
        return x,False
    return x,success and True

def num_list(args:tuple[str,bool]):
    x = args[0]
    success = args[1] if len(args) > 1 else True
    if len(x) < 2 or x[0] != "[" or x[-1] != "]":
        return x,False
    var_contents = x[1:-1]
    list_contents = var_contents.split(',')
    for expected_number in list_contents:
        _,result = greater_than_zero((expected_number,))
        if not result:
            return x,False
    return x,success and True

def parse_coord_list(x:str):
    data = loads(x)
    return list(map(lambda coord:(coord["x"],coord["y"]),data))

def parse_num_list(x:str):
    nums = x[1:-1].split(',')
    return [int(num_as_str) for num_as_str in nums]

def parse_env():
    validators:Dict[str,Callable[[str,bool],tuple[str,bool]]] = {}
    validators['ENVIRONMENT_X_AXIS'] = greater_than_zero
    validators['ENVIRONMENT_Y_AXIS'] = greater_than_zero
    validators['INITIAL_TURN_ON_PROBABILITY'] = greater_than_zero
    validators['OBSTACLES_COORDS'] = coord_list
    validators['POINTS_OF_INTEREST_COORDS'] = coord_list
    validators['POINTS_OF_INTEREST_VISIT_TIMES'] = num_list
    validators['START_X_COORD'] = non_negative_integer
    validators['START_Y_COORD'] = non_negative_integer
    validators['TOTAL_TIME'] = greater_than_zero
    validators['UAV_BATTERY'] = greater_than_zero
    validators['UAV_AMOUNT'] = greater_than_zero
    validators['UAV_CHARGE_TIME'] = greater_than_zero
    for var_name,validator in validators.items():
        var_value = getenv(var_name)
        if var_value is None:
            raise ValueError(f"Env var {var_name} missing")
        _,var_is_valid = validator((var_value,))
        if not var_is_valid:
            raise ValueError(f"Env var {var_name} is not valid")

# Check which environment to load
PY_ENV = getenv('PY_ENV')
if PY_ENV is None:
    raise ValueError('PY_ENV is not specified')
if not file_exists(f"./envs/.env.{PY_ENV}"):
    raise TypeError(f"env \"{PY_ENV}\" does not exist")
print(f"Loading .env.{PY_ENV}")

# Load environment
load_dotenv(dotenv_path=f"./envs/.env.{PY_ENV}")

parse_env()
ENVIRONMENT_X_AXIS = int(getenv('ENVIRONMENT_X_AXIS'))
ENVIRONMENT_Y_AXIS = int(getenv('ENVIRONMENT_Y_AXIS'))
INITIAL_TURN_ON_PROBABILITY = int(getenv('INITIAL_TURN_ON_PROBABILITY'))
OBSTACLES_COORDS = parse_coord_list(getenv('OBSTACLES_COORDS'))
POINTS_OF_INTEREST_COORDS = parse_coord_list(getenv('POINTS_OF_INTEREST_COORDS'))
POINTS_OF_INTEREST_VISIT_TIMES = parse_num_list(getenv('POINTS_OF_INTEREST_VISIT_TIMES'))
START_X_COORD = int(getenv('START_X_COORD'))
START_Y_COORD = int(getenv('START_Y_COORD'))
TOTAL_TIME = int(getenv('TOTAL_TIME'))
UAV_AMOUNT = int(getenv('UAV_AMOUNT'))
UAV_BATTERY = int(getenv('UAV_BATTERY'))
UAV_CHARGE_TIME = int(getenv('UAV_CHARGE_TIME'))