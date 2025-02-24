from flask import Flask

app = Flask(__name__)

from video.blueprint import bp as video
from image.blueprint import bp as image
from text.blueprint import bp as text

app.register_blueprint(video)
app.register_blueprint(image)
app.register_blueprint(text)


if __name__ == "__main__":
    # run single-threaded so we don't run out of Vram with multiple requests
    app.run(debug=True, threaded=False)
