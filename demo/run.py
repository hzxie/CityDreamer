# -*- coding: utf-8 -*-
#
# @File:   run.py
# @Author: Haozhe Xie
# @Date:   2023-06-30 10:12:55
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-07-06 20:20:06
# @Email:  root@haozhexie.com

import argparse
import flask
import numpy as np
import io
import logging
import os
import sys
import uuid

from PIL import Image

DEMO_HOME_DIR = os.path.abspath(os.path.dirname(__file__))
# Add parent dir to PYTHONPATH
sys.path.append(os.path.join(DEMO_HOME_DIR, os.path.pardir))

import scripts.inference as inference
import scripts.dataset_generator as inference_helper

CONSTANTS = {"UPLOAD_DIR": "/tmp"}
MODELS = {
    "vqae": None,
    "sampler": None,
    "gancraft_bg": None,
    "gancraft_fg": None,
}

# The Flask application
app = flask.Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    with open(os.path.join(DEMO_HOME_DIR, "index.html")) as f:
        return f.read()


@app.route("/img/normalize.action", methods=["POST"])
def normalize_image():
    img = Image.open(io.BytesIO(flask.request.files.get("image").read()))
    img = np.array(img, dtype=np.float32)
    scale = np.max(img) / 255.0
    img /= scale
    file_name = "%s.png" % uuid.uuid4()

    Image.fromarray(img.astype(np.uint8)).save(
        os.path.join(CONSTANTS["UPLOAD_DIR"], file_name)
    )
    return flask.jsonify({"scale": scale, "filename": file_name})


@app.route("/img/get-normalized/<file_name>", methods=["GET"])
def get_normalized_image(file_name):
    file_path = os.path.join(CONSTANTS["UPLOAD_DIR"], file_name)
    if not os.path.exists(file_path):
        flask.abort(404)

    return flask.send_file(file_path, mimetype="image/png")


def get_runtime_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampler_ckpt",
        default=os.path.join(inference.PROJECT_HOME, "output", "sampler.pth"),
    )
    parser.add_argument(
        "--gancraft_bg_ckpt",
        default=os.path.join(inference.PROJECT_HOME, "output", "gancraft-bg.pth"),
    )
    parser.add_argument(
        "--gancraft_fg_ckpt",
        default=os.path.join(inference.PROJECT_HOME, "output", "gancraft-fg.pth"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    args = get_runtime_arguments()
    # logging.info("Initialize models ...")
    # (
    #     MODELS["vqae"],
    #     MODELS["sampler"],
    #     MODELS["gancraft_bg"],
    #     MODELS["gancraft_fg"],
    # ) = inference.get_models(
    #     args.sampler_ckpt, args.gancraft_bg_ckpt, args.gancraft_fg_ckpt
    # )

    # Start HTTP server
    logging.info("Launching demo ...")
    app.run(host="0.0.0.0", port=8080, debug=True)
