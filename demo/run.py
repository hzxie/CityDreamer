# -*- coding: utf-8 -*-
#
# @File:   run.py
# @Author: Haozhe Xie
# @Date:   2023-06-30 10:12:55
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-07-14 16:27:28
# @Email:  root@haozhexie.com

import argparse
import flask
import json
import numpy as np
import io
import logging
import os
import sys
import torch
import uuid

from PIL import Image

DEMO_HOME_DIR = os.path.abspath(os.path.dirname(__file__))
# Add parent dir to PYTHONPATH
sys.path.append(os.path.join(DEMO_HOME_DIR, os.path.pardir))

import scripts.inference as inference
import scripts.dataset_generator as inference_helper
import utils.helpers

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


@app.route("/image/upload.action", methods=["POST"])
def upload_image():
    img = Image.open(io.BytesIO(flask.request.files.get("image").read()))

    file_name = "%s.png" % uuid.uuid4()
    img.save(os.path.join(CONSTANTS["UPLOAD_DIR"], file_name))
    return flask.jsonify({"filename": file_name})


@app.route("/image/<file_name>", methods=["GET"])
def get_image(file_name):
    file_path = os.path.join(CONSTANTS["UPLOAD_DIR"], file_name)
    if not os.path.exists(file_path):
        flask.abort(404)

    return flask.send_file(file_path, mimetype="image/png")


@app.route("/image/<file_name>/normalize.action", methods=["GET"])
def normalize_image(file_name):
    file_path = os.path.join(CONSTANTS["UPLOAD_DIR"], file_name)
    img = np.array(Image.open(file_path), dtype=np.float32)
    scale = np.max(img) / 255.0
    img /= scale

    file_name = "%s.png" % uuid.uuid4()
    Image.fromarray(img.astype(np.uint8)).save(
        os.path.join(CONSTANTS["UPLOAD_DIR"], file_name)
    )
    return flask.jsonify({"filename": file_name})


@app.route("/video/<file_name>", methods=["GET"])
def get_video(file_name):
    file_path = os.path.join(CONSTANTS["UPLOAD_DIR"], file_name)
    print(file_path, os.path.exists(file_path))
    if not os.path.exists(file_path):
        flask.abort(404)

    return flask.send_file(file_path, mimetype="video/mp4")


@app.route("/trajectory/preview.action", methods=["POST"])
def get_trajectory_preview():
    hf_filename = flask.request.form.get("hf")
    hf_filepath = os.path.join(CONSTANTS["UPLOAD_DIR"], hf_filename)
    seg_filename = flask.request.form.get("seg")
    seg_filepath = os.path.join(CONSTANTS["UPLOAD_DIR"], seg_filename)
    trajectory = flask.request.form.get("trajectory")

    if not os.path.exists(hf_filepath) or not os.path.exists(seg_filepath):
        flask.abort(404)
    try:
        trajectory = json.loads(trajectory)
    except Exception as ex:
        logging.exception(ex)
        flask.abort(400)

    hf = np.array(Image.open(hf_filepath)).astype(np.int32)
    seg = np.array(Image.open(seg_filepath).convert("P")).astype(np.int32)
    output_file = os.path.join(CONSTANTS["UPLOAD_DIR"], "%s.mp4" % uuid.uuid4())
    frames = []
    for t in trajectory:
        frames.append(_get_seg_volume_rendering(hf, seg, t))

    inference.get_video(frames, output_file)
    return flask.jsonify({"filename": os.path.basename(output_file)})


def _get_seg_volume_rendering(hf, seg, pos):
    tx, ty = int(pos["target"]["x"]), int(pos["target"]["y"])
    part_hf, part_seg = inference.get_part_hf_seg(hf, seg, tx, ty)
    seg_volume = inference_helper.get_seg_volume(part_seg, part_hf)
    (
        voxel_id,
        depth2,
        raydirs,
        cam_origin,
    ) = inference.get_voxel_intersection_perspective(
        seg_volume,
        {
            "x": pos["camera"]["x"] - tx + inference.CONSTANTS["LAYOUT_VOL_SIZE"] // 2,
            "y": pos["camera"]["y"] - ty + inference.CONSTANTS["LAYOUT_VOL_SIZE"] // 2,
            "z": pos["camera"]["z"],
        },
    )
    seg_map = utils.helpers.get_seg_map(voxel_id.squeeze()[..., 0].cpu().numpy())
    frame = inference_helper._get_diffuse_shading_img(
        seg_map,
        depth2.squeeze(dim=0).permute(2, 0, 1, 3, 4),
        raydirs.squeeze(dim=0),
        cam_origin.squeeze(dim=0),
    )
    return np.array(frame)[..., ::-1]


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
