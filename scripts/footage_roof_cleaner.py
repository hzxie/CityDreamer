# -*- coding: utf-8 -*-
#
# @File:   footage_roof_cleaner.py
# @Author: Haozhe Xie
# @Date:   2023-07-05 09:59:13
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-07-06 09:02:50
# @Email:  root@haozhexie.com

import argparse
import cv2
import logging
import multiprocessing
import numpy as np
import os
import pickle
import random

from PIL import Image
from tqdm import tqdm

# Disable the warning message for PIL decompression bomb
# Ref: https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None

# CONSTANTS
CONSTANTS = {
    "BLD_INS_LABEL_MIN": 10,
    "LAYOUT_VOL_SIZE": 1536,
}


def downsample_texture(roof_img_file):
    texture = cv2.imread(roof_img_file)
    texture = cv2.GaussianBlur(texture.astype(np.uint8), (0, 0), sigmaX=3, sigmaY=3)
    texture = cv2.resize(
        np.tile(texture, (48, 48, 1)), (texture.shape[1], texture.shape[0])
    )
    texture = np.tile(texture, (8, 8, 1))
    return texture


def main(ges_dir, roof_img_dir, city="US-NewYork"):
    roof_textures = [
        cv2.imread(os.path.join(roof_img_dir, rf)) for rf in os.listdir(roof_img_dir)
    ]
    roof_files = [
        os.path.join(roof_img_dir, rf)
        for rf in os.listdir(roof_img_dir)
    ]
    with multiprocessing.Pool() as pool:
        roof_textures = list(
            tqdm(
                pool.imap(downsample_texture, roof_files),
                total=len(roof_files),
                desc="Preparing roof texture...",
            )
        )

    ges_projects = sorted([gp for gp in os.listdir(ges_dir) if gp.startswith(city)])
    for gp in tqdm(ges_projects, desc="Repainting roof..."):
        footage_dir = os.path.join(ges_dir, gp, "footage")
        raycasting_dir = os.path.join(ges_dir, gp, "raycasting")
        if not os.path.exists(raycasting_dir):
            logging.warning(
                "Skip Project %s. No raycasting found in %s" % (gp, raycasting_dir)
            )
            continue

        footages = sorted(os.listdir(footage_dir))
        raycastings = sorted(os.listdir(raycasting_dir))
        for f, r in zip(footages, raycastings):
            footage = cv2.imread(os.path.join(footage_dir, f))
            with open(os.path.join(raycasting_dir, r), "rb") as fp:
                raycasting = pickle.load(fp)

            bev_seg_map = raycasting["voxel_id"][:, :, None, 0, 0]
            # Assume the ID of a facade instance is 2k, the corresponding roof instance is 2k - 1.
            instances = [i for i in np.unique(bev_seg_map) if i > 10 and i % 2 == 0]

            for i in instances:
                roof_mask = bev_seg_map == i - 1
                mx, my, mw, mh = cv2.boundingRect(
                    cv2.findNonZero(roof_mask.astype(np.uint8))
                )

                roof_texture = random.choice(roof_textures).copy()
                rh, rw, _ = roof_texture.shape
                # Apply random affine transformation
                rot_mtrx = cv2.getRotationMatrix2D(
                    (rw // 2, rh // 2), random.randint(0, 90), 1
                )
                roof_texture = cv2.warpAffine(
                    roof_texture, rot_mtrx, (rh, rw), borderMode=cv2.BORDER_REPLICATE
                )
                rh, rw, _ = roof_texture.shape
                ry, rx = rh // 2 - mh // 2, rw // 2 - mw // 2
                assert rh > mh and rw > mw, (rh, mh, rw, mw)
                roof_img = np.zeros(footage.shape, dtype=np.uint8)
                roof_img[my : my + mh, mx : mx + mw] = roof_texture[
                    ry : ry + mh, rx : rx + mw
                ]
                # Make the border of roof more smooth
                roof_mask = cv2.GaussianBlur(
                    roof_mask.astype(np.uint8), (0, 0), sigmaX=3, sigmaY=3
                )
                roof_mask = cv2.dilate(roof_mask, np.ones((7, 7), dtype=np.uint8))
                roof_mask = cv2.erode(roof_mask, np.ones((7, 7), dtype=np.uint8))
                roof_mask = cv2.GaussianBlur(roof_mask, (0, 0), sigmaX=1, sigmaY=1)
                roof_mask = roof_mask[..., None]
                # Replace roof textures
                footage = footage * (1 - roof_mask) + roof_img * roof_mask

            footage = cv2.imwrite(
                os.path.join(footage_dir, f), footage.astype(np.uint8)
            )


if __name__ == "__main__":
    PROJECT_HOME = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir)
    )
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--ges_dir", default=os.path.join(PROJECT_HOME, "data", "ges"))
    parser.add_argument(
        "--roof_img_dir", default=os.path.join(PROJECT_HOME, "output", "roofs")
    )
    args = parser.parse_args()
    main(args.ges_dir, args.roof_img_dir)
