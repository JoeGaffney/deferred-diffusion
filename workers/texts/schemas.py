from typing import List, Literal

from pydantic import Base64Bytes, BaseModel, Field, RootModel


class TextContent(BaseModel):
    type: str = "text"
    text: str


class MessageContent(BaseModel):
    type: str
    text: str = ""


class MessageItem(BaseModel):
    role: str
    content: List[MessageContent]


class TextRequest(BaseModel):
    temperature: float = 0.7
    seed: int = 42
    model: Literal["Qwen/Qwen2.5-VL-3B-Instruct", "gpt-4o-mini", "gpt-4.1-mini"] = Field(
        description="model", default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    messages: list[MessageItem] = Field(description="List of messages", default=[])
    images: List[str] = Field(description="Image references", default=[])
    videos: List[str] = Field(description="Video references", default=[])


class TextWorkerResponse(BaseModel):
    response: str
    chain_of_thought: list
