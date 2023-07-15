# -*- coding: utf-8 -*-
#
# @File:   run.py
# @Author: Haozhe Xie
# @Date:   2023-06-30 10:12:55
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-07-15 16:43:29
# @Email:  root@haozhexie.com

import argparse
import cv2
import flask
import flask_executor
import json
import numpy as np
import io
import logging
import os
import sys
import uuid

from PIL import Image
from tqdm import tqdm

DEMO_HOME_DIR = os.path.abspath(os.path.dirname(__file__))
# Add parent dir to PYTHONPATH
sys.path.append(os.path.join(DEMO_HOME_DIR, os.path.pardir))

import scripts.inference as inference
import utils.helpers

CONSTANTS = {
    "GES_IMAGE_HEIGHT": 540,
    "GES_IMAGE_WIDTH": 960,
}
MODELS = {
    "vqae": None,
    "sampler": None,
    "gancraft_bg": None,
    "gancraft_fg": None,
}

# The Flask application
app = flask.Flask(__name__)
app.config["EXECUTOR_PROPAGATE_EXCEPTIONS"] = True
executor = flask_executor.Executor(app)


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


@app.route("/image/<video_name>/<frame_id>", methods=["GET"])
def get_vide_frame(video_name, frame_id):
    frame_id = int(frame_id) if frame_id.isdigit() else 0
    file_path = os.path.join(CONSTANTS["UPLOAD_DIR"], video_name, "%04d.jpg" % frame_id)
    if not os.path.exists(file_path):
        flask.abort(404)

    return flask.send_file(file_path, mimetype="image/jpg")


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

    output_file = os.path.join(CONSTANTS["UPLOAD_DIR"], "%s.mp4" % uuid.uuid4())
    frames = get_seg_volume_rendering(hf_filepath, seg_filepath, trajectory)
    inference.get_video(frames, output_file)
    return flask.jsonify({"filename": os.path.basename(output_file)})


@app.route("/city/render.action", methods=["POST"])
def render():
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

    video_name = "%s" % uuid.uuid4()
    output_dir = os.path.join(CONSTANTS["UPLOAD_DIR"], video_name)
    os.makedirs(output_dir, exist_ok=True)
    executor.submit(
        get_city_rendering, hf_filepath, seg_filepath, trajectory, output_dir
    )
    return flask.jsonify({"video": video_name, "frames": len(trajectory)})


def get_seg_volume_rendering(hf, seg, trajectory):
    # Invoked by get_trajectory_preview()
    hf = np.array(Image.open(hf)).astype(np.int32)
    seg = np.array(Image.open(seg).convert("P")).astype(np.int32)
    frames = []
    for t in tqdm(trajectory, desc="Rendering seg volume"):
        tx, ty = int(t["target"]["x"]), int(t["target"]["y"])
        part_hf, part_seg = inference.get_part_hf_seg(
            hf, seg, tx, ty, inference.CONSTANTS["LAYOUT_VOL_SIZE"]
        )
        seg_volume = inference.get_seg_volume(part_hf, part_seg)
        (
            voxel_id,
            depth2,
            raydirs,
            cam_origin,
        ) = inference.get_voxel_intersection_perspective(
            seg_volume,
            {
                "x": t["camera"]["x"]
                - tx
                + inference.CONSTANTS["LAYOUT_VOL_SIZE"] // 2,
                "y": t["camera"]["y"]
                - ty
                + inference.CONSTANTS["LAYOUT_VOL_SIZE"] // 2,
                "z": t["camera"]["z"],
            },
        )
        seg_map = utils.helpers.get_seg_map(voxel_id.squeeze()[..., 0].cpu().numpy())
        frame = utils.helpers.get_diffuse_shading_img(
            seg_map,
            depth2.squeeze(dim=0).permute(2, 0, 1, 3, 4),
            raydirs.squeeze(dim=0),
            cam_origin.squeeze(dim=0),
        )
        frames.append(np.array(frame)[..., ::-1])

    return frames


def get_city_rendering(hf, seg, trajectory, output_dir):
    hf = np.array(Image.open(hf)).astype(np.int32)
    seg = np.array(Image.open(seg).convert("P"))
    seg, building_stats = inference.get_instance_seg_map(seg)
    seg = seg.astype(np.int32)

    bg_z, building_zs = inference.get_latent_codes(
        building_stats,
        MODELS["gancraft_bg"].module.cfg.NETWORK.GANCRAFT.STYLE_DIM,
        MODELS["gancraft_bg"].output_device,
    )
    for f_idx, t in enumerate(tqdm(trajectory, desc="Rendering city")):
        tx, ty = int(t["target"]["x"]), int(t["target"]["y"])
        part_hf, part_seg = inference.get_part_hf_seg(
            hf, seg, tx, ty, inference.CONSTANTS["EXTENDED_VOL_SIZE"]
        )
        _building_stats = inference.get_part_building_stats(
            part_seg, building_stats, tx, ty
        )
        seg_volume = inference.get_seg_volume(part_hf, part_seg)
        hf_seg = inference.get_hf_seg_tensor(
            part_hf, part_seg, MODELS["gancraft_bg"].output_device
        )
        img = inference.render(
            (CONSTANTS["PATCH_HEIGHT"], CONSTANTS["PATCH_WIDTH"]),
            seg_volume,
            hf_seg,
            {
                "x": t["camera"]["x"]
                - tx
                + inference.CONSTANTS["LAYOUT_VOL_SIZE"] // 2,
                "y": t["camera"]["y"]
                - ty
                + inference.CONSTANTS["LAYOUT_VOL_SIZE"] // 2,
                "z": t["camera"]["z"],
            },
            MODELS["gancraft_bg"],
            MODELS["gancraft_fg"],
            _building_stats,
            bg_z,
            building_zs,
        )
        img = (utils.helpers.tensor_to_image(img, "RGB") * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, "%04d.jpg" % f_idx), img[..., ::-1])


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
    parser.add_argument(
        "--upload_dir",
        default="/tmp/city-dreamer",
    )
    parser.add_argument(
        "--patch_height",
        default=CONSTANTS["GES_IMAGE_HEIGHT"] // 5,
        type=int,
    )
    parser.add_argument(
        "--patch_width",
        default=CONSTANTS["GES_IMAGE_WIDTH"] // 5,
        type=int,
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    args = get_runtime_arguments()

    # Register runtime arguments to global variables
    os.makedirs(args.upload_dir, exist_ok=True)
    CONSTANTS["UPLOAD_DIR"] = args.upload_dir
    CONSTANTS["PATCH_HEIGHT"] = args.patch_height
    CONSTANTS["PATCH_WIDTH"] = args.patch_width

    # Initialize models
    logging.info("Initialize models ...")
    (
        MODELS["vqae"],
        MODELS["sampler"],
        MODELS["gancraft_bg"],
        MODELS["gancraft_fg"],
    ) = inference.get_models(
        args.sampler_ckpt, args.gancraft_bg_ckpt, args.gancraft_fg_ckpt
    )

    # Start HTTP server
    logging.info("Launching demo ...")
    app.run(host="0.0.0.0", port=3186, debug=True)
