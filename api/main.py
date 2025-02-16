from flask import Flask
from common.context import Context

app = Flask(__name__)

from img_to_video.blueprint import bp as img_to_video
from img_to_img.blueprint import bp as img_to_img
from text_to_img.blueprint import bp as text_to_img

app.register_blueprint(img_to_video)
app.register_blueprint(img_to_img)
app.register_blueprint(text_to_img)


if __name__ == "__main__":
    # run single-threaded so we don't run out of Vram with multiple requests
    app.run(debug=True, threaded=False)
