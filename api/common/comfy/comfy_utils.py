import inspect
from typing import Any, Dict, List, get_args, get_origin

from pydantic import BaseModel


def model_schema_to_comfy_nodes(schema: BaseModel, start_id=1) -> Dict[str, Any]:
    """Convert a Pydantic model schema to a complete ComfyUI workflow structure."""
    nodes = []
    node_id = start_id
    last_node_id = start_id

    # Calculate positions - simple vertical layout
    x_pos = 0
    y_pos = 0
    y_spacing = 150  # Vertical spacing between nodes

    # In Pydantic v2, use model_fields instead of __fields__
    for name, field in schema.model_fields.items():
        # Get the field type from the annotation
        ftype = field.annotation
        default = field.default if field.default is not None else ""

        # Skip complex types or lists
        if get_origin(ftype) is list:
            continue
        if isinstance(default, BaseModel):
            continue

        # Map python types to Comfy node classes
        if ftype is int or ftype == int:
            class_type = "PrimitiveInt"
        elif ftype is float or ftype == float:
            class_type = "PrimitiveFloat"
        elif ftype is str or ftype == str:
            # Use multiline for strings
            class_type = "PrimitiveStringMultiline"
        else:
            # Skip unsupported types
            continue

        # Create node in the format ComfyUI expects
        node = {
            "id": node_id,
            "type": class_type,
            "pos": [x_pos, y_pos],
            "size": [400, 50],
            "flags": {},
            "order": node_id - start_id,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "title": f"api_{name}",
            "properties": {},
            "widgets_values": [default],
        }

        nodes.append(node)
        last_node_id = node_id
        node_id += 1
        y_pos += y_spacing  # Move position for next node

    # Create the complete workflow structure
    workflow = {
        "last_node_id": last_node_id,
        "last_link_id": 1,
        "nodes": nodes,
        "links": [],
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }

    return workflow
