# -*- coding: utf-8 -*-
#
# @File:   dataset_generator.py
# @Author: Haozhe Xie
# @Date:   2023-03-21 18:26:26
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-03-26 19:01:08
# @Email:  root@haozhexie.com

import argparse
import cv2
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
            {"k": "highway", "v": "trunk"},
            {"k": "highway", "v": "trunk_link"},
            {"k": "highway", "v": "primary"},
            {"k": "highway", "v": "primary_link"},
            {"k": "highway", "v": "secondary"},
            {"k": "highway", "v": "secondary_link"},
            {"k": "highway", "v": "tertiary"},
            {"k": "highway", "v": "motorway"},
            {"k": "highway", "v": "service"},
            {"k": "highway", "v": "residential"},
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
    coastlines, nodes = osm_helper.get_highways(
        osm_file_path, [{"k": "natural", "v": "coastline"}], nodes
    )
    nodes = osm_helper.get_nodes_lng_lat(osm_file_path, nodes)
    return highways, footprints, coastlines, nodes


def _tag_equals(tags, key, values=None):
    if key not in tags:
        return False
    if values is None:
        return True
    return tags[key] in values


def _get_highway_color(map_name, highway_tags):
    if map_name == "height_field":
        return 0
    elif map_name == "seg_map":
        if _tag_equals(highway_tags, "natural", ["coastline"]):
            return 5
        else:
            return 1
    else:
        raise Exception("Unknown map name: %s" % map_name)


def _get_footprint_color(map_name, footprint_tags):
    if map_name == "height_field":
        if _tag_equals(footprint_tags, "role", ["inner"]):
            # "role" in footprint_tags and footprint_tags["role"] == "inner"
            return None
        elif _tag_equals(footprint_tags, "building:levels"):
            # "building:levels" in footprint_tags:
            return int(float(footprint_tags["building:levels"]) * 4.26)
        elif _tag_equals(footprint_tags, "building", ["roof"]):
            #  "building" in footprint_tags and footprint_tags["building"] == "roof"
            return None
        elif _tag_equals(footprint_tags, "leisure", ["park", "grass", "garden"]):
            # "leisure" in footprint_tags and footprint_tags["leisure"] in [...]
            return 5
        elif _tag_equals(footprint_tags, "landuse", ["construction"]):
            #  "building" in footprint_tags and footprint_tags["building"] == "construction"
            return 10
        else:
            assert "height" in footprint_tags
            return int(float(footprint_tags["height"]) + 0.5)
    elif map_name == "seg_map":
        if _tag_equals(footprint_tags, "role", ["inner"]):
            return 2
        elif _tag_equals(footprint_tags, "building"):
            return 2
        elif _tag_equals(footprint_tags, "leisure", ["park", "grass", "garden"]):
            return 3
        elif _tag_equals(footprint_tags, "landuse", ["construction"]):
            return 4
        elif (
            _tag_equals(footprint_tags, "waterway")
            or _tag_equals(footprint_tags, "water")
            or _tag_equals(footprint_tags, "natural", ["water"])
            or _tag_equals(footprint_tags, "leisure", ["marina"])
        ):
            return 5
        else:
            return 0
    else:
        raise Exception("Unknown map name: %s" % map_name)


def get_coast_zones(coastlines, seg_map):
    N_PIXELS_THRES = 1e5
    coastlines = cv2.dilate((seg_map == 5).astype(np.uint8), np.ones((10, 10)))
    seg_map[coastlines != 0] = 5
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (seg_map == 0).astype(np.uint8)
    )
    coast_zones = np.zeros_like(seg_map).astype(bool)
    for i in range(n_labels):
        if stats[i][-1] > N_PIXELS_THRES:
            logging.debug("Mask[ID=%d] contains %d pixels." % (i, stats[i][-1]))
            coast_zones[labels == i] = True

    coast_zones[seg_map != 0] = False
    coast_zones[coastlines != 0] = True
    return coast_zones


def get_osm_images(osm_file_path, zoom_level):
    logging.debug("Reading OSM files ...")
    highways, footprints, coastlines, nodes = get_highways_and_footprints(osm_file_path)

    logging.debug("Converting lng/lat to X/Y coordinates ...")
    lnglat_bounds = {
        k: float(v) for k, v in osm_helper.get_lnglat_bounds(osm_file_path).items()
    }
    resolution = osm_helper.get_map_resolution(lnglat_bounds, zoom_level)
    nodes = osm_helper.get_nodes_xy_coordinates(nodes, resolution, zoom_level)
    xy_bounds = osm_helper.get_xy_bounds(nodes)

    # Fix missing height (for buildings) and width (for highways and coastlines)
    highways = osm_helper.fix_missing_highway_width(highways)
    coastlines = osm_helper.fix_missing_highway_width(coastlines)
    footprints = osm_helper.fix_missing_footprint_height(
        footprints, osm_helper.get_footprint_height_stat(footprints)
    )

    # Generate semantic labels
    seg_map = osm_helper.get_empty_map(xy_bounds)
    seg_map = osm_helper.plot_highways(
        "seg_map", _get_highway_color, seg_map, highways, nodes, xy_bounds, resolution
    )
    seg_map = osm_helper.plot_highways(
        "seg_map", _get_highway_color, seg_map, coastlines, nodes, xy_bounds, resolution
    )
    seg_map = osm_helper.plot_footprints(
        "seg_map",
        _get_footprint_color,
        seg_map,
        footprints,
        nodes,
        xy_bounds,
    )
    coast_zones = get_coast_zones(coastlines, seg_map.copy())
    seg_map[coast_zones] = 5
    # Assign ID=6 to unlabelled pixels (regarded as ground)
    seg_map[seg_map == 0] = 6

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
    # Make sure that all height values are above 0
    height_field[coast_zones] = -5
    height_field += 5
    height_field = (height_field * resolution).astype(np.uint16)

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
