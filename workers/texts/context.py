from texts.schemas import ModelName, TextRequest


class TextContext:
    def __init__(self, model: ModelName, data: TextRequest):
        self.model = model
        self.data = data
