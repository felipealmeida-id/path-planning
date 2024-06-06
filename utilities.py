from collections import Counter
from enums import ProgramModules


def prepare_directory(module_enum: ProgramModules):
    from env_parser import Env
    env = Env.get_instance()
    root = f"./output/{env.PY_ENV}"
    required_folders = [
        f"./output/{env.PY_ENV}/gan/generator",
        f"./output/{env.PY_ENV}/gan/discriminator",
        f"./output/{env.PY_ENV}/gan/generated_imgs",
        f"./output/{env.PY_ENV}/pather/generated_paths",
        f"./inputs/{env.PY_ENV}/",
        f"./output/profiling/{env.PY_ENV}"
    ]
    if(module_enum == ProgramModules.PERCEPTRON):
        delete_existing_results(root+"/gan")
    create_necessary_folders(required_folders)
    # unzip_inputs()


def unzip_inputs():
    from zipfile import ZipFile
    from os import listdir 
    from env_parser import Env
    env = Env.get_instance()
    if len(listdir(f'inputs/{env.PY_ENV}')) == 0:
        with ZipFile(f'./zip_inputs/{env.PY_ENV}.zip','r') as zip_ref:
            zip_ref.extractall(f'inputs/{env.PY_ENV}')

def delete_existing_results(root:str):
    from shutil import rmtree
    from os.path import exists
    if exists(root):
        rmtree(root)

def create_necessary_folders(required_paths):
    from os import makedirs
    for folder_path in required_paths:
        if not folders_exist(folder_path):
            makedirs(folder_path)

def folders_exist(folders_path:str):
    from os import path
    folders = folders_path.split(path.sep)
    current_path = ""
    for folder in folders:
        current_path = path.join(current_path, folder)
        if not path.exists(current_path) or not path.isdir(current_path):
            return False
    return True

def find_enum_by_value(enum_cls, value):
    for enum_member in enum_cls:
        if enum_member.value == value:
            return enum_member
    raise ValueError(f"No enum member with value '{value}' found for class {enum_cls.__name__}.")

def get_duplicates(array):
    c = Counter(array)
    return {k: v for k, v in c.items() if v > 1}