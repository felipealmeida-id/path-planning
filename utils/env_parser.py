from dotenv import load_dotenv
from typing import Dict,Callable
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
    if len(x) < 2 or x[0] != "[" or x[-1] != "]":
        return x,False
    x = x[1:-1]
    try:
        eval(x)
    except SyntaxError:
        return x,False
    return x,success and True

def parse_env():
    validators:Dict[str,Callable[[str,bool],tuple[str,bool]]] = {}
    validators['ENVIRONMENT_X_AXIS'] = greater_than_zero
    validators['ENVIRONMENT_Y_AXIS'] = greater_than_zero
    validators['INITIAL_TURN_ON_PROBABILITY'] = greater_than_zero
    validators['POINTS_OF_INTEREST_AMOUNT'] = greater_than_zero
    validators['POINTS_OF_INTEREST_COORDS'] = coord_list
    validators['TOTAL_TIME'] = greater_than_zero
    validators['UAV_BATTERY'] = greater_than_zero
    validators['UAV_AMOUNT'] = greater_than_zero
    for var_name,validator in validators.items():
        var_value = getenv(var_name)
        if var_value is None:
            raise ValueError(f"Env var {var_name} missing")
        _,var_is_valid = validator((var_value,))
        if not var_is_valid:
            raise ValueError(f"Env var {var_name} is not valid")

# Check which environment to load
# PY_ENV = getenv('PY_ENV')
# if PY_ENV is None:
#     raise ValueError('PY_ENV is not specified')
# if not file_exists(f"./envs/.env.{PY_ENV}"):
#     raise KeyError(f"env \"{PY_ENV}\" does not exist")
# print(f"Loading .env.{PY_ENV}")

# Load environment
# load_dotenv(dotenv_path=f"./envs/.env.{PY_ENV}")
load_dotenv(dotenv_path=f"./envs/.env.local")

parse_env()
ENVIRONMENT_X_AXIS = int(getenv('ENVIRONMENT_X_AXIS'))
ENVIRONMENT_Y_AXIS = int(getenv('ENVIRONMENT_Y_AXIS'))
POINTS_OF_INTEREST_AMOUNT = int(getenv('POINTS_OF_INTEREST_AMOUNT'))
INITIAL_TURN_ON_PROBABILITY = int(getenv('INITIAL_TURN_ON_PROBABILITY'))
TOTAL_TIME = int(getenv('TOTAL_TIME'))
UAV_AMOUNT = int(getenv('UAV_AMOUNT'))
UAV_BATTERY = int(getenv('UAV_BATTERY'))