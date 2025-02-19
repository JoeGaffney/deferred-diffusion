from flask import Flask
from common.context import Context

app = Flask(__name__)

from video.blueprint import bp as video
from image.blueprint import bp as image

app.register_blueprint(video)
app.register_blueprint(image)


if __name__ == "__main__":
    # run single-threaded so we don't run out of Vram with multiple requests
    app.run(debug=True, threaded=False)
