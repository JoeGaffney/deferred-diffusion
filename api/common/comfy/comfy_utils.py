import inspect
from typing import Any, Dict, get_args, get_origin

from pydantic import BaseModel


def model_schema_to_comfy_nodes(schema: BaseModel, start_id=100) -> Dict[str, Any]:
    nodes = {}
    node_id = start_id

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
            class_type = "PrimitiveString"
        else:
            # Skip unsupported types
            continue

        nodes[str(node_id)] = {
            "class_type": class_type,
            "inputs": {"value": default},
            "_meta": {"title": f"api_{name}"},
        }
        node_id += 1

    return nodes
