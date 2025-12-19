from typing import Any, Dict, List, Literal, Optional, TypeAlias
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, ConfigDict, Field, model_validator

from common.schemas import TaskStatus

ClassTypes: TypeAlias = Literal[
    "PrimitiveInt",
    "PrimitiveFloat",
    "PrimitiveStringMultiline",
    "LoadImage",
    "LoadVideo",
]

Workflow: TypeAlias = Dict[str, Dict[str, Any]]


class Patch(BaseModel):
    title: str  # matches node["_meta"]["title"]
    class_type: ClassTypes
    value: Any

    @model_validator(mode="after")
    def _validate(self):
        if self.class_type == "PrimitiveInt" and not isinstance(self.value, int):
            raise ValueError("PrimitiveInt value must be int")
        if self.class_type == "PrimitiveFloat" and not isinstance(self.value, (int, float)):
            raise ValueError("PrimitiveFloat value must be number")
        if self.class_type == "PrimitiveStringMultiline" and not isinstance(self.value, str):
            raise ValueError("PrimitiveStringMultiline value must be string")
        if self.class_type in ("LoadImage", "LoadVideo") and not isinstance(self.value, str):
            raise ValueError("LoadImage/LoadVideo value must be base64 string")
        return self


class WorkflowRequest(BaseModel):
    workflow: Workflow
    patches: List[Patch]

    @model_validator(mode="after")
    def _validate_patches(self):
        title_to_ids: dict[str, list[str]] = {}
        for node_id, node in self.workflow.items():
            if isinstance(node, dict):
                title = node.get("_meta", {}).get("title")
                if isinstance(title, str):
                    title_to_ids.setdefault(title, []).append(node_id)

        for patch in self.patches:
            ids = title_to_ids.get(patch.title, [])
            if not ids:
                raise ValueError(f"Patch title '{patch.title}' not found in workflow")
            if len(ids) > 1:
                raise ValueError(f"Patch title '{patch.title}' is not unique (matched {ids})")

            node = self.workflow[ids[0]]
            if node.get("class_type") != patch.class_type:
                raise ValueError(
                    f"Patch title '{patch.title}' expected class_type={patch.class_type} "
                    f"but workflow has {node.get('class_type')}"
                )

        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workflow": {
                    "1": {
                        "inputs": {"value": "A mountain range"},
                        "class_type": "PrimitiveStringMultiline",
                        "_meta": {"title": "positive_prompt"},
                    },
                    "2": {"inputs": {"value": 1024}, "class_type": "PrimitiveInt", "_meta": {"title": "width"}},
                    "3": {
                        "inputs": {"ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"},
                        "class_type": "CheckpointLoaderSimple",
                        "_meta": {"title": "Load Checkpoint"},
                    },
                    "4": "...",
                },
                "patches": [
                    {
                        "title": "positive_prompt",
                        "class_type": "PrimitiveStringMultiline",
                        "value": "A snowy mountain range at sunset",
                    },
                    {"title": "width", "class_type": "PrimitiveInt", "value": 2048},
                ],
            }
        }
    )


class WorkflowOutput(BaseModel):
    data_type: Literal["image", "video"]
    filename: str
    base64_data: Base64Bytes


class WorkflowWorkerResponse(BaseModel):
    logs: List[str] = []
    outputs: List[WorkflowOutput]


class WorkflowResponse(BaseModel):
    id: UUID
    status: TaskStatus
    result: Optional[WorkflowWorkerResponse] = None
    error_message: Optional[str] = None
    logs: List[str] = []
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "SUCCESS",
                "result": {
                    "outputs": [
                        {
                            "data_type": "image",
                            "base64_data": "iVBORw0KGgoAAAANSUhEUgAA...",
                            "filename": "comfy_node_filename.png",
                        }
                    ],
                },
                "error_message": None,
                "logs": ["Setup", "Progress: 10%", "Progress: 20%", "..."],
            }
        }
    )


class WorkflowCreateResponse(BaseModel):
    id: UUID
    status: TaskStatus
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "PENDING",
            }
        }
    )
