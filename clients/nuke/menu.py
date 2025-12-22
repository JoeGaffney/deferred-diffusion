import dd_image
import dd_workflow

# Add custom nodes to the Nuke node menu
nuke.menu("Nodes").addCommand("DeferredDiffusion/DDImage", "nuke.createNode('dd_image')")
nuke.menu("Nodes").addCommand("DeferredDiffusion/DDVideo", "nuke.createNode('dd_video')")
nuke.menu("Nodes").addCommand("DeferredDiffusion/DDText", "nuke.createNode('dd_text')")
nuke.menu("Nodes").addCommand("DeferredDiffusion/DDWorkflow", "nuke.createNode('dd_workflow')")
