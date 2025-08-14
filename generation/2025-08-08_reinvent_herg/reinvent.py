import os
import importlib
import inspect

components_path = "/home/corey/miniconda3/envs/reinvent4/lib/python3.10/site-packages/reinvent_plugins/components"  # Replace with your absolute path

print("Component modules and classes:")

for filename in os.listdir(components_path):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = f"reinvent_plugins.components.{filename[:-3]}"
        try:
            module = importlib.import_module(module_name)
            classes = [name for name, obj in inspect.getmembers(module) if inspect.isclass(obj)]
            print(f"{module_name}: {classes}")
        except Exception as e:
            print(f"Failed to import {module_name}: {e}")
