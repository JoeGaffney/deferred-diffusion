from texts.schemas import ModelName, TextRequest


class TextContext:
    def __init__(self, data: TextRequest):
        self.model = data.model
        self.data = data
