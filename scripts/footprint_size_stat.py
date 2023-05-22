# -*- coding: utf-8 -*-
#
# @File:   footprint_size_stat.py
# @Author: Haozhe Xie
# @Date:   2023-05-22 11:47:32
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-05-22 20:10:33
# @Email:  root@haozhexie.com

import argparse
import cv2
import logging
import os
import numpy as np

from PIL import Image
from tqdm import tqdm

# Disable the warning message for PIL decompression bomb
# Ref: https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None


def _get_building_instance_stat(seg_map, contours):
    BULIDING_MASK_ID = 2
    N_PIXELS_THRES = 16
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (1 - contours).astype(np.uint8), connectivity=4
    )
    # Remove non-building instance masks
    labels[seg_map != BULIDING_MASK_ID] = 0
    # Remove too small buildings
    ignored_indexes = np.where(stats[:, -1] <= N_PIXELS_THRES)[0]
    labels[np.isin(labels, ignored_indexes)] = 0

    remaining_instances = np.unique(labels)
    assert 0 in remaining_instances
    remaining_instances = remaining_instances[1:]
    assert 0 not in remaining_instances
    # NOTE: assert stats.shape[1] == 5, represents x, y, w, h, area of the components.
    return stats[remaining_instances]


def main(osm_dir, selected_city):
    cities = os.listdir(osm_dir)
    for city in tqdm(cities):
        if selected_city is not None and city != selected_city:
            continue

        hf_file_path = os.path.join(osm_dir, city, "hf.png")
        ctr_file_path = os.path.join(osm_dir, city, "ctr.png")
        seg_map_file_path = os.path.join(osm_dir, city, "seg.png")
        height_field = np.array(Image.open(hf_file_path))
        contours = np.array(Image.open(ctr_file_path))
        seg_map = np.array(Image.open(seg_map_file_path).convert("P"))

        stats = _get_building_instance_stat(seg_map, contours)
        logging.info("Max Building Height: %.2f" % np.max(height_field))
        logging.info(
            "Max Building Size (WxH): (%.2fx%.2f)"
            % (np.max(stats[:, 2]), np.max(stats[:, 3]))
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    PROJECT_HOME = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir)
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--osm_dir", default=os.path.join(PROJECT_HOME, "data", "osm"))
    parser.add_argument("--city", default="US-NewYork")
    args = parser.parse_args()
    main(args.osm_dir, args.city)
