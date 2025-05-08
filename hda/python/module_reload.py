import importlib
import os
import sys

import hou


def reload_modules_from_path(path):
    abs_path = os.path.abspath(path)
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue  # Skip built-ins or dynamically created modules
        module_path = os.path.abspath(module_file)
        if module_path.startswith(abs_path):
            print(f"Reloading: {name}")
            importlib.reload(module)


# Get the current node (the HDA node)
node_type = "deferred_diffusion::Cop/image"
hda = hou.nodeType(node_type).definition().libraryFilePath()
hda_python = os.path.dirname(hda) + "/python"

# Add the Python path if it's not already in sys.path
if hda_python not in sys.path:
    sys.path.append(hda_python)
    print("Python path appended: " + hda_python)

# Reload all modules from this directory
reload_modules_from_path(hda_python)
