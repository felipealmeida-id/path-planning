from dotenv import load_dotenv
from typing import Dict,Callable
from json import loads
from os import getenv
from os.path import exists as file_exists
from torch import device,cuda

class Env:
    __instance = None
    BATCH_SIZE:int
    DEVICE:str
    D_LEARN_RATE:float
    ENVIRONMENT_X_AXIS:int
    ENVIRONMENT_Y_AXIS:int
    EPOCHS:int
    G_LEARN_RATE:float
    INITIAL_TURN_ON_PROBABILITY:int
    K:int
    NOISE_DIMENSION:int
    OBSTACLES_COORDS:list
    POINTS_OF_INTEREST_COORDS:list
    POINTS_OF_INTEREST_VISIT_TIMES:list[int]
    PY_ENV:str
    SAMPLE_SIZE:int
    SAMPLES_TO_GENERATE:int
    START_X_COORD:int
    START_Y_COORD:int
    TOTAL_TIME:int
    UAV_AMOUNT:int
    UAV_BATTERY:int
    UAV_CHARGE_TIME:int

    def __init__(self):
        # Check which environment to load
        PY_ENV = getenv('PY_ENV')
        MODULE = getenv('MODULE')
        if PY_ENV is None:
            raise ValueError('PY_ENV is not specified')
        if not file_exists(f"./envs/.env.{PY_ENV}"):
            raise TypeError(f"env \"{PY_ENV}\" does not exist")
        print(f"Loading .env.{PY_ENV}")
        load_dotenv(dotenv_path=f"./envs/.env.{PY_ENV}")
        switch_dict = {
            "pather":self._parse_env_pather,
            "gan":self._parse_env_gan
        }
        switch_dict[MODULE]()
        self.BATCH_SIZE = int(getenv('BATCH_SIZE'))
        self.DEVICE = device('cuda' if cuda.is_available() else 'cpu')
        self.D_LEARN_RATE = float(getenv('D_LEARN_RATE'))
        self.ENVIRONMENT_X_AXIS = int(getenv('ENVIRONMENT_X_AXIS'))
        self.ENVIRONMENT_Y_AXIS = int(getenv('ENVIRONMENT_Y_AXIS'))
        self.EPOCHS = int(getenv('EPOCHS'))
        self.G_LEARN_RATE = float(getenv('G_LEARN_RATE'))
        self.INITIAL_TURN_ON_PROBABILITY = int(getenv('INITIAL_TURN_ON_PROBABILITY'))
        self.K = int(getenv('K'))
        self.NOISE_DIMENSION = int(getenv('NOISE_DIMENSION'))
        self.OBSTACLES_COORDS = self._parse_coord_list(getenv('OBSTACLES_COORDS'))
        self.POINTS_OF_INTEREST_COORDS = self._parse_coord_list(getenv('POINTS_OF_INTEREST_COORDS'))
        self.POINTS_OF_INTEREST_VISIT_TIMES = self._parse_num_list(getenv('POINTS_OF_INTEREST_VISIT_TIMES'))
        self.PY_ENV = PY_ENV
        self.SAMPLE_SIZE = int(getenv('SAMPLE_SIZE'))
        self.SAMPLES_TO_GENERATE = int(getenv('SAMPLES_TO_GENERATE'))
        self.START_X_COORD = int(getenv('START_X_COORD'))
        self.START_Y_COORD = int(getenv('START_Y_COORD'))
        self.TOTAL_TIME = int(getenv('TOTAL_TIME'))
        self.UAV_AMOUNT = int(getenv('UAV_AMOUNT'))
        self.UAV_BATTERY = int(getenv('UAV_BATTERY'))
        self.UAV_CHARGE_TIME = int(getenv('UAV_CHARGE_TIME'))

    @classmethod
    def get_instance(self):
        if(Env.__instance is None):
            Env.__instance = Env()
        return Env.__instance

    def _positive_float(self,args:tuple[str,bool]):
        x:str = args[0]
        success:bool = args[1] if len(args) > 1 else True
        try:
            x_as_float = float(x)
            return x,success and x_as_float > 0 
        except:
            return x,False

    def _non_negative_integer(self,args:tuple[str,bool]):
        x:str = args[0]
        success:bool = args[1] if len(args) > 1 else True
        return x,(success and x.isdigit())

    def _non_positive_integer(self,args:tuple[str,bool]):
        x:str = args[0]
        success:bool = args[1] if len(args) > 1 else True
        return x,(success and x[0]=='-' and x[1:].isdigit())

    def _non_zero(self,args:tuple[str,bool]):
        x:str = args[0]
        success:bool = args[1] if len(args) > 1 else True
        return x,(success and x!='0')

    def _greater_than_zero_int(self,args:tuple[str,bool]):
        x = args[0]
        success = args[1] if len(args) > 1 else True
        return self._non_zero(self._non_negative_integer((x,success)))

    def _coord_list(self,args:tuple[str,bool]):
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

    def _num_list(self,args:tuple[str,bool]):
        x = args[0]
        success = args[1] if len(args) > 1 else True
        if len(x) < 2 or x[0] != "[" or x[-1] != "]":
            return x,False
        var_contents = x[1:-1]
        list_contents = var_contents.split(',')
        for expected_number in list_contents:
            _,result = self._greater_than_zero_int((expected_number,))
            if not result:
                return x,False
        return x,success and True

    def _parse_coord_list(self,x:str):
        data = loads(x)
        return list(map(lambda coord:(coord["x"],coord["y"]),data))

    def _parse_num_list(self,x:str):
        nums = x[1:-1].split(',')
        return [int(num_as_str) for num_as_str in nums]

    def _validate_parameters(self,validators:Dict[str,Callable[[str,bool],tuple[str,bool]]]):
        for var_name,validator in validators.items():
            var_value = getenv(var_name)
            if var_value is None:
                raise ValueError(f"Env var {var_name} missing")
            _,var_is_valid = validator((var_value,))
            if not var_is_valid:
                raise ValueError(f"Env var {var_name} is not valid")

    def _parse_env_pather(self):
        validators:Dict[str,Callable[[str,bool],tuple[str,bool]]] = {}
        validators['ENVIRONMENT_X_AXIS'] = self._greater_than_zero_int
        validators['ENVIRONMENT_Y_AXIS'] = self._greater_than_zero_int
        validators['INITIAL_TURN_ON_PROBABILITY'] = self._greater_than_zero_int
        validators['SAMPLES_TO_GENERATE'] = self._greater_than_zero_int
        validators['OBSTACLES_COORDS'] = self._coord_list
        validators['POINTS_OF_INTEREST_COORDS'] = self._coord_list
        validators['POINTS_OF_INTEREST_VISIT_TIMES'] = self._num_list
        validators['START_X_COORD'] = self._non_negative_integer
        validators['START_Y_COORD'] = self._non_negative_integer
        validators['TOTAL_TIME'] = self._greater_than_zero_int
        validators['UAV_BATTERY'] = self._greater_than_zero_int
        validators['UAV_AMOUNT'] = self._greater_than_zero_int
        validators['UAV_CHARGE_TIME'] = self._greater_than_zero_int
        self._validate_parameters(validators)

    def _parse_env_gan(self):
        validators:Dict[str,Callable[[str,bool],tuple[str,bool]]] = {}
        validators['BATCH_SIZE'] = self._greater_than_zero_int
        validators['D_LEARN_RATE'] = self._positive_float
        validators['EPOCHS'] = self._greater_than_zero_int
        validators['G_LEARN_RATE'] = self._positive_float
        validators['K'] = self._greater_than_zero_int
        validators['NOISE_DIMENSION'] = self._greater_than_zero_int
        validators['SAMPLE_SIZE'] = self._greater_than_zero_int
        self._validate_parameters(validators)