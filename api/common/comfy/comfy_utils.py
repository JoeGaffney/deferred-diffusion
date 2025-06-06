from typing import Any, List, Literal, Tuple, Union, get_args, get_origin

from pydantic import BaseModel

from common.schemas import ComfyWorkflowNode, ComfyWorkflowResponse


def is_string_literal(ftype: Any) -> bool:
    origin = get_origin(ftype)
    if origin is Literal:
        args = get_args(ftype)
        return all(isinstance(arg, str) for arg in args)
    return False


def is_string_type(ftype: Any) -> bool:
    if ftype is str or ftype == str:
        return True

    # Check for Optional[str] (Union[str, None])
    origin = get_origin(ftype)
    if origin is not None and origin is Union:
        args = get_args(ftype)
        return len(args) == 2 and str in args and (type(None) in args or None in args)

    return False


def is_image_field(field_type: Any, field_extra) -> bool:
    """Check if a field is an image field based on its type and metadata."""
    if not field_extra or not is_string_type(field_type):
        return False

    if callable(field_extra):
        return False

    content_encoding = field_extra.get("contentEncoding")
    content_type = field_extra.get("contentMediaType")

    return content_encoding == "base64" and content_type is not None and "image/" in str(content_type)


def process_schema_nodes(
    schema: BaseModel, node_id: int = 1, position: List[int] = [0, 0], prefix: str = "", max_list_items: int = 2
) -> Tuple[List[ComfyWorkflowNode], int, List[int]]:
    """
    Recursively process a schema and generate appropriate nodes.

    Args:
        schema: The Pydantic model to process
        node_id: Starting node ID
        position: [x, y] position for the first node
        prefix: Prefix for node titles
        max_list_items: Maximum number of items to create for list fields

    Returns:
        Tuple containing:
        - List of generated nodes
        - The next available node ID
        - The updated position for the next node
    """
    nodes = []
    x_pos, y_pos = position
    y_spacing = 75
    default_size = [400, 100]
    image_size = [400, 300]

    # Process each field in the schema
    for name, field in schema.model_fields.items():
        field_type = field.annotation
        default = field.default if field.default is not None else ""
        field_extra = field.json_schema_extra if hasattr(field, "json_schema_extra") else None

        # Handle list types
        if get_origin(field_type) is list:
            inner_type = get_args(field_type)[0]

            # We only care about lists of BaseModel types
            if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                for i in range(max_list_items):
                    item_prefix = f"{name}_{i}_"
                    full_prefix = f"{prefix}{item_prefix}" if prefix else item_prefix

                    # Create a dummy instance and process it
                    dummy_instance = inner_type.model_construct()
                    inner_nodes, node_id, new_position = process_schema_nodes(
                        dummy_instance, node_id, [x_pos, y_pos], full_prefix, max_list_items
                    )
                    nodes.extend(inner_nodes)
                    # Update y_pos from the returned position
                    x_pos, y_pos = new_position
            continue

        # Skip complex types that aren't BaseModels
        if isinstance(default, BaseModel):
            continue

        # Create node based on type
        title_prefix = f"api_{prefix}" if prefix else "api_"
        title = f"{title_prefix}{name}"
        size = default_size

        if is_image_field(field_type, field_extra):
            class_type = "LoadImage"
            widgets_values = ["", "image"]
            size = image_size
        elif field_type is int or field_type == int:
            class_type = "PrimitiveInt"
            widgets_values = [default, "fixed"]
        elif field_type is float or field_type == float:
            class_type = "PrimitiveFloat"
            widgets_values = [default]
        elif is_string_type(field_type):
            class_type = "PrimitiveString"
            if field_extra:
                if not callable(field_extra):
                    format_type = field_extra.get("format")
                    if format_type == "multi_line":
                        class_type = "PrimitiveStringMultiline"

            widgets_values = [default]
        elif is_string_literal(field_type):
            class_type = "PrimitiveString"
            widgets_values = [""]
        else:
            # Skip unsupported types
            continue

        # Create the node
        node = ComfyWorkflowNode(
            id=node_id,
            type=class_type,
            pos=[x_pos, y_pos],
            size=size,
            flags={},
            order=node_id,
            mode=0,
            inputs=[],
            outputs=[],
            title=title,
            properties={},
            widgets_values=widgets_values,
        )

        nodes.append(node)
        node_id += 1
        y_pos += y_spacing + size[1]  # Move position for next node

    return nodes, node_id, [x_pos, y_pos]


def model_schema_to_comfy_nodes(schema: BaseModel, start_id=1, max_list_items=2) -> ComfyWorkflowResponse:
    """Convert a Pydantic model schema to a complete ComfyUI workflow structure."""
    # Process the schema recursively
    nodes, last_node_id, _ = process_schema_nodes(schema, node_id=start_id, max_list_items=max_list_items)

    # Create the complete workflow structure
    workflow = ComfyWorkflowResponse(
        last_node_id=last_node_id,
        last_link_id=1,
        nodes=nodes,
    )

    return workflow
