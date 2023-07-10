from dotenv import load_dotenv
from typing import Dict,Callable

def non_negative_integer(x:str,success=True):
    return x,(success and x.isdigit())

def non_positive_integer(x:str,success=True):
    return x,(success and x[0]=='-' and x[1:].isdigit())

def non_zero(x:str,success=True):
    return x,(success and x!=0)

def parse_env():
    load_dotenv()
    validators:Dict[str,Callable[[str,bool],bool]] = {}
    validators['UAV_BATTERY'] = non_positive_integer
