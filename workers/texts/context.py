from texts.schemas import ModelName, TextRequest


class TextContext:
    def __init__(self, data: TextRequest):
        self.model: ModelName = data.model
        self.data = data
