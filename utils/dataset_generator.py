# -*- coding: utf-8 -*-
#
# @File:   dataset_generator.py
# @Author: Haozhe Xie
# @Date:   2023-03-21 18:26:26
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-03-23 15:21:33
# @Email:  root@haozhexie.com

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

import osm_helper

from tqdm import tqdm
from PIL import Image


def get_highways_and_footprints(osm_file_path):
    highways, nodes = osm_helper.get_highways(
        osm_file_path,
        [
            "trunk",
            "trunk_link",
            "primary",
            "primary_link",
            "secondary",
            "secondary_link",
            "tertiary",
            "motorway",
            "service",
            "residential",
        ],
    )
    footprints, nodes = osm_helper.get_footprints(
        osm_file_path,
        [
            "building",
            {"k": "landuse", "v": ["construction"]},
            {"k": "leisure", "v": ["park", "marina"]},
        ],
        nodes,
    )
    nodes = osm_helper.get_nodes_lng_lat(osm_file_path, nodes)
    return highways, footprints, nodes


def _get_highway_color(map_name, highway_tags):
    if map_name == "height_field":
        return 0
    elif map_name == "seg_map":
        return 1
    else:
        raise Exception("Unknown map name: %s" % map_name)


def _get_footprint_color(map_name, footprint_tags):
    if map_name == "height_field":
        if "role" in footprint_tags and footprint_tags["role"] == "inner":
            return None
        if "building:levels" in footprint_tags:
            return int(float(footprint_tags["building:levels"]) * 4.26)
        elif "building" in footprint_tags and footprint_tags["building"] == "roof":
            return None
        elif "leisure" in footprint_tags and footprint_tags["leisure"] in [
            "park",
            "grass",
            "garden",
        ]:
            return 5
        elif (
            "landuse" in footprint_tags and footprint_tags["landuse"] == "construction"
        ):
            return 10
        else:
            assert "height" in footprint_tags
            return int(float(footprint_tags["height"]) + 0.5)
    elif map_name == "seg_map":
        if "role" in footprint_tags and footprint_tags["role"] == "inner":
            return 2
        elif "building" in footprint_tags:
            return 2
        elif "leisure" in footprint_tags and footprint_tags["leisure"] in [
            "park",
            "grass",
            "garden",
        ]:
            return 3
        elif (
            "landuse" in footprint_tags and footprint_tags["landuse"] == "construction"
        ):
            return 4
        elif "leisure" in footprint_tags and footprint_tags["leisure"] == "marina":
            return 5
        else:
            return 0
    else:
        raise Exception("Unknown map name: %s" % map_name)


def get_osm_images(osm_file_path, zoom_level):
    logging.debug("Reading OSM files ...")
    highways, footprints, nodes = get_highways_and_footprints(osm_file_path)

    logging.debug("Converting lng/lat to X/Y coordinates ...")
    lnglat_bounds = {
        k: float(v) for k, v in osm_helper.get_lnglat_bounds(osm_file_path).items()
    }
    resolution = osm_helper.get_map_resolution(lnglat_bounds, zoom_level)
    nodes = osm_helper.get_nodes_xy_coordinates(nodes, resolution, zoom_level)
    xy_bounds = osm_helper.get_xy_bounds(nodes)

    # Fix missing height (for buildings) and width (for highways)
    highways = osm_helper.fix_missing_highway_width(highways)
    footprints = osm_helper.fix_missing_footprint_height(
        footprints, osm_helper.get_footprint_height_stat(footprints)
    )

    # Generate height fields
    height_field = osm_helper.get_empty_map(xy_bounds)
    height_field = osm_helper.plot_footprints(
        "height_field",
        _get_footprint_color,
        height_field,
        footprints,
        nodes,
        xy_bounds,
    )
    ## TODO: Consider coastlines
    height_field += 5
    # TODO: Remove normalization
    # height_field = (height_field * resolution).astype(np.uint16)
    height_field = (height_field / np.max(height_field) * 255).astype(np.uint8)

    # Generate semantic labels
    seg_map = osm_helper.get_empty_map(xy_bounds)
    seg_map = osm_helper.plot_highways(
        "seg_map", _get_highway_color, seg_map, highways, nodes, xy_bounds, resolution
    )
    seg_map = osm_helper.plot_footprints(
        "seg_map",
        _get_footprint_color,
        seg_map,
        footprints,
        nodes,
        xy_bounds,
    )
    return height_field, seg_map


def get_seg_map_img(seg_map):
    PALETTE = np.array([[i, i, i] for i in range(256)])
    PALETTE[:16] = np.array(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [191, 0, 0],
            [64, 128, 0],
            [191, 128, 0],
            [64, 0, 128],
            [191, 0, 128],
            [64, 128, 128],
            [191, 128, 128],
        ]
    )
    seg_map = Image.fromarray(seg_map.astype(np.uint8))
    seg_map.putpalette(PALETTE.reshape(-1).tolist())
    return seg_map


def main(osm_dir, zoom_level):
    osm_files = os.listdir(osm_dir)
    for of in tqdm(osm_files):
        basename, suffix = os.path.splitext(of)
        if suffix != ".osm":
            continue
        height_field, seg_map = get_osm_images(os.path.join(osm_dir, of), zoom_level)
        Image.fromarray(height_field).save(
            os.path.join(osm_dir, "%s-hf.png" % basename)
        )
        get_seg_map_img(seg_map).save(os.path.join(osm_dir, "%s-seg.png" % basename))
        # TODO: Remove break
        break


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (48, 30)
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--osm_dir", dest="osm_dir", default="../data/osm")
    parser.add_argument("--zoom", dest="zoom", default=18)
    args = parser.parse_args()
    main(args.osm_dir, args.zoom)
