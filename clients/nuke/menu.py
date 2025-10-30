import dd_image

# Add custom nodes to the Nuke node menu
nuke.menu("Nodes").addCommand("DeferredDiffusion/DDImage", "nuke.createNode('dd_image')")
nuke.menu("Nodes").addCommand("DeferredDiffusion/DDImageExternal", "nuke.createNode('dd_imageExternal')")
nuke.menu("Nodes").addCommand("DeferredDiffusion/DDReference", "nuke.createNode('dd_reference')")
nuke.menu("Nodes").addCommand("DeferredDiffusion/DDVideo", "nuke.createNode('dd_video')")
nuke.menu("Nodes").addCommand("DeferredDiffusion/DDText", "nuke.createNode('dd_text')")
