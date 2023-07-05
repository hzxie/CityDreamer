# -*- coding: utf-8 -*-
#
# @File:   footage_asphalt_cleaner.py
# @Author: Haozhe Xie
# @Date:   2023-07-03 09:49:53
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-07-05 11:02:52
# @Email:  root@haozhexie.com

import argparse
import cv2
import logging
import numpy as np
import pickle
import os

from tqdm import tqdm


def main(ges_dir, asphalt_img):
    asphalt_img = cv2.imread(asphalt_img)
    ah, aw, _ = asphalt_img.shape
    ges_projects = os.listdir(ges_dir)
    for gp in tqdm(ges_projects, leave=True):
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
            fh, fw, _ = footage.shape
            with open(os.path.join(raycasting_dir, r), "rb") as fp:
                raycasting = pickle.load(fp)

            # ROAD_ID == 1
            road_mask = raycasting["voxel_id"][:, :, None, 0, 0] == 1
            y, x = np.random.randint(0, ah - fh), np.random.randint(0, aw - fw)
            _asphalt_img = asphalt_img[y : y + fh, x : x + fw]
            footage = _asphalt_img * road_mask + footage * (1 - road_mask)
            footage = cv2.imwrite(
                os.path.join(footage_dir, f), footage.astype(np.uint8)
            )
            # Update the mask for roads
            raycasting["mask"][road_mask[..., 0]] = 1
            with open(os.path.join(raycasting_dir, r), "wb") as fp:
                pickle.dump(raycasting, fp)


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
        "--asphalt_img", default=os.path.join(PROJECT_HOME, "output", "asphalt.jpg")
    )
    args = parser.parse_args()
    main(args.ges_dir, args.asphalt_img)
