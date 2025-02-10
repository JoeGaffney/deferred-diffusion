from flask import Flask
from common.context import Context


from flask import Blueprint, request, jsonify

app = Flask(__name__)
from blueprints.diffusion import bp as diffusion_bp

app.register_blueprint(diffusion_bp)


if __name__ == "__main__":
    app.run(debug=True)

# stable_video_diffusion.py
# Ensure this file contains the necessary imports and definitions for `main` and `Context`
