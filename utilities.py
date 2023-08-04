from collections import Counter

def prepare_directory():
    from env_parser import Env
    env = Env.get_instance()
    root = f"./output/{env.PY_ENV}"
    required_folders = [
        "gan/generator",
        "gan/discriminator",
        "gan/generated_imgs",
        "pather/generated_paths"
    ]
    required_paths = map(lambda path: f"{root}/{path}",required_folders)
    delete_existing_results(root)
    create_necessary_folders(required_paths)

def delete_existing_results(root:str):
    from shutil import rmtree
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