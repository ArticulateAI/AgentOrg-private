import os
import importlib

# Dynamically import all Python files in the current directory
module_dir = os.path.dirname(__file__)

for file in os.listdir(module_dir):
    if file.endswith(".py") and file != "__init__.py":
        module_name = file[:-3]  # Strip off '.py' to get the module name
        importlib.import_module(f'.{module_name}', package=__name__)