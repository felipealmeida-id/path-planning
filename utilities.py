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
