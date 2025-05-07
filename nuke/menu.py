import dd_image

# Add custom nodes to the Nuke node menu
nuke.menu("Nodes").addCommand("DeferredDiffusion/DDImage", "nuke.createNode('dd_image')")
nuke.menu("Nodes").addCommand("DeferredDiffusion/DDControlnet", "nuke.createNode('dd_controlnet')")
nuke.menu("Nodes").addCommand("DeferredDiffusion/DDAdapter", "nuke.createNode('dd_adapter')")
