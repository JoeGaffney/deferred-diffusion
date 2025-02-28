from text.schemas import TextRequest


class TextContext:
    def __init__(self, data: TextRequest):
        self.data = data
