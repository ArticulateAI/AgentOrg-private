import importlib
import os
import pkgutil
from pathlib import Path

# Automatically import all Python files in the current directory and subdirectories
def import_all_modules_from_package(package_name: str):
    package_path = Path(__file__).parent
    for module_info in pkgutil.walk_packages(
        path=[str(package_path)],
        prefix=f"{package_name}."
    ):
        if not module_info.ispkg:  # Skip directories
            importlib.import_module(module_info.name)

# Dynamically import all modules under workers.tools
import_all_modules_from_package("agentorg.env.tools")

# Expose TOOL_REGISTRY
from .tools import TOOL_REGISTRY