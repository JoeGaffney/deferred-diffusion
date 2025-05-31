import inspect
from typing import Any, Dict, List, Optional, get_args, get_origin

from pydantic import BaseModel


def model_schema_to_comfy_nodes(schema: BaseModel, start_id=1) -> Dict[str, Any]:
    """Convert a Pydantic model schema to a complete ComfyUI workflow structure."""
    nodes = []
    node_id = start_id
    last_node_id = start_id

    # Calculate positions - simple vertical layout
    x_pos = 0
    y_pos = 0
    y_spacing = 50  # Vertical spacing between nodes

    # Default node properties
    default_size = [400, 50]
    image_size = [400, 300]  # Size for image nodes
    title_prefix = "api_"

    # In Pydantic v2, use model_fields instead of __fields__
    for name, field in schema.model_fields.items():
        # Get the field type from the annotation
        ftype = field.annotation
        print(f"Processing field: {name}, type: {ftype}")
        default = field.default if field.default is not None else ""

        # Check if this is an image field by examining json_schema_extra
        is_image = False
        if hasattr(field, "json_schema_extra") and field.json_schema_extra:
            content_encoding = field.json_schema_extra.get("contentEncoding")
            content_type = field.json_schema_extra.get("contentMediaType")
            if content_encoding == "base64" and content_type and "image/" in str(content_type):
                is_image = True

        # Skip complex types or lists
        if get_origin(ftype) is list:
            continue
        if isinstance(default, BaseModel):
            continue

        size = default_size
        title = f"{title_prefix}{name}"

        # Set node properties based on type
        if is_image:
            class_type = "LoadImage"
            widgets_values = ["", "image"]
            size = image_size
        elif ftype is int or ftype == int:
            class_type = "PrimitiveInt"
            widgets_values = [default]
        elif ftype is float or ftype == float:
            class_type = "PrimitiveFloat"
            widgets_values = [default]
        elif ftype is str or ftype == str:
            # Use multiline for strings
            class_type = "PrimitiveStringMultiline"
            widgets_values = [default]
        else:
            # Skip unsupported types
            continue

        # Create node with common structure
        node = {
            "id": node_id,
            "type": class_type,
            "pos": [x_pos, y_pos],
            "size": size,
            "flags": {},
            "order": node_id - start_id,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "title": title,
            "properties": {},
            "widgets_values": widgets_values,
        }

        nodes.append(node)
        last_node_id = node_id
        node_id += 1
        y_pos += y_spacing + size[1]  # Move position for next node

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
