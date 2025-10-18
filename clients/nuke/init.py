import os
import sys

import nuke

# Get base path dynamically (where this file lives)
BASE_DIR = os.path.dirname(__file__)

# Plugin folders
# Add gizmo folder
nuke.pluginAddPath(os.path.join(BASE_DIR, "gizmos"))

# Add Python paths (shared deps, generated code)
sys.path.append(os.path.join(BASE_DIR, "python"))
sys.path.append(os.path.join(BASE_DIR, "python", "generated"))
