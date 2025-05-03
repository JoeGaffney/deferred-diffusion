from typing import List

from pydantic import Base64Bytes, BaseModel, Field


class TextResponse(BaseModel):
    response: str
    chain_of_thought: list


class TextRequest(BaseModel):
    temperature: float = 0.7
    seed: int = 42
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    messages: list
    images: List[Base64Bytes] = Field(description="Image references", default=[])
    videos: List[Base64Bytes] = Field(description="Video references", default=[])
