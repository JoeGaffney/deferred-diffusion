import nuke


def create_dd_image_node():
    # Create the node from the gizmo (no need to re-define defaults)
    node = nuke.createNode("dd_image")  # 'dd_image' is the name of your gizmo

    # Optionally: You can set other properties or interact with the node here
    # e.g., If you want to call a function defined inside the gizmo, you can do it here

    return node


def process_image(node):
    # This function is defined to process the node
    nuke.message("Processing image...")
    nuke.tprint(node.name())
    nuke.tprint(node)

    # Example of manipulating the node's properties
    print("node", node)
    print(f"Processing image for node: {node.name()}")
    # Example of manipulating the node's properties
