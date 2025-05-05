import dd_image

# Add custom nodes to the Nuke node menu
nuke.menu("Nodes").addCommand("DeferredDiffusion/DDImage", "nuke.createNode('dd_image')")
