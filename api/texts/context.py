from texts.schemas import TextRequest


class TextContext:
    def __init__(self, data: TextRequest):
        self.data = data
        self.model = data.model
