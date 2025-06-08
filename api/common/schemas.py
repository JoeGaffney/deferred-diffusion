from typing import Any, Dict, List, Optional, Tuple, Union, get_args, get_origin

from pydantic import BaseModel, Field


class ComfyWorkflowNode(BaseModel):
    id: int
    type: str
    pos: List[int]
    size: List[int]
    flags: Dict[str, Any] = Field(default_factory=dict)
    order: int
    mode: int = 0
    inputs: List[Dict[str, Any]] = Field(default_factory=list)
    outputs: List[Dict[str, Any]] = Field(default_factory=list)
    title: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    widgets_values: List[Any]


class ComfyWorkflowResponse(BaseModel):
    last_node_id: int
    last_link_id: int
    nodes: List[ComfyWorkflowNode]
    links: List[Dict[str, Any]] = Field(default_factory=list)
    groups: List[Dict[str, Any]] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)
    version: float = 0.4
