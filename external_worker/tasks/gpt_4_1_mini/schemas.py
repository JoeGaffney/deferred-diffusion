from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: str


class Gpt41MiniParams(BaseModel):
    messages: List[ChatMessage] = Field(...)
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)

    max_tokens: Optional[int] = Field(default=None)

    max_tokens: Optional[int] = Field(default=None)

    max_tokens: Optional[int] = Field(default=None)

    max_tokens: Optional[int] = Field(default=None)
