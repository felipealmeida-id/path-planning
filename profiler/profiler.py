from cProfile import Profile
from pstats import Stats
from csv import DictWriter
from typing import Callable
from env_parser import Env
import os

class CustomProfiler:
    def __init__(self):
        self.env = Env.get_instance()

    def profile_fun(self,fun_to_prof:Callable[[],None]):
        pr = Profile()
        pr.enable()
        fun_to_prof()
        pr.disable()
        ps = Stats(pr)
        file_name = self.determine_file_unique_name(f"output/profiling/{self.env.PY_ENV}/profile_data.csv")
        with open(file_name, "w", newline="") as csvfile:
            fieldnames = ["function_name", "ncalls", "tottime", "cumtime"]
            writer = DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for func, info in ps.stats.items():
                writer.writerow(
                    {
                        "function_name": func,
                        "ncalls": info[0],
                        "tottime": info[2],
                        "cumtime": info[3],
                    }
                )

    def determine_file_unique_name(self,filename:str):
        base_name, extension = os.path.splitext(filename)
        new_filename = filename
        counter = 1
        while os.path.exists(new_filename):
            new_filename = f"{base_name}_{counter}{extension}"
            counter += 1
        return new_filename