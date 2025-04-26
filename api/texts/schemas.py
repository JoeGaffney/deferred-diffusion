from pydantic import BaseModel


class TextResponse(BaseModel):
    response: str
    chain_of_thought: list


class TextRequest(BaseModel):
    temperature: float = 0.7
    seed: int = 42
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    messages: list
