from dotenv import load_dotenv
from typing import Dict,Callable
from os import getenv

def non_negative_integer(x:str,success=True):
    return x,(success and x.isdigit())

def non_positive_integer(x:str,success=True):
    return x,(success and x[0]=='-' and x[1:].isdigit())

def non_zero(x:str,success=True):
    return x,(success and x!='0')

def greater_than_zero(x:str,success=True):
    return non_zero(non_negative_integer(x,success))

def parse_env():
    validators:Dict[str,Callable[[str,bool],bool]] = {}
    validators['ENVIRONMENT_X_AXIS'] = greater_than_zero
    validators['ENVIRONMENT_Y_AXIS'] = greater_than_zero
    validators['TOTAL_TIME'] = greater_than_zero
    validators['UAV_BATTERY'] = greater_than_zero
    validators['UAV_AMOUNT'] = greater_than_zero 

load_dotenv()
ENVIRONMENT_X_AXIS = int(getenv('ENVIRONMENT_X_AXIS'))
ENVIRONMENT_Y_AXIS = int(getenv('ENVIRONMENT_Y_AXIS'))
TOTAL_TIME = int(getenv('TOTAL_TIME'))
UAV_AMOUNT = int(getenv('UAV_AMOUNT'))
UAV_BATTERY = int(getenv('UAV_BATTERY'))