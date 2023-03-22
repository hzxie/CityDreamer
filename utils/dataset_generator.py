# -*- coding: utf-8 -*-
#
# @File:   dataset_generator.py
# @Author: Haozhe Xie
# @Date:   2023-03-21 18:26:26
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-03-22 20:52:54
# @Email:  root@haozhexie.com

import argparse
import logging
import matplotlib.pyplot as plt
import os

import osm_helper

from tqdm import tqdm


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
    if map_name == "hf":
        return 5
    else:
        raise Exception("Unknown map name: %s" % map_name)


def _get_footprint_color(map_name, footprint_tags):
    if map_name == "hf":
        assert "height" in footprint_tags
        return int(float(footprint_tags["height"]) + 0.5)
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
    height_field = osm_helper.plot_highways(
        "hf",
        _get_highway_color,
        height_field,
        highways,
        nodes,
        xy_bounds,
        resolution
    )
    height_field = osm_helper.plot_footprints(
        "hf",
        _get_footprint_color,
        height_field,
        footprints,
        nodes,
        xy_bounds,
    )
    # TODO: Consider the resolution
    plt.imshow(height_field)
    plt.savefig(osm_file_path.replace(".osm", "-Preview.jpg"))

    # Generate semantic labels

    # Generate instance labels


def main(osm_dir, zoom_level):
    osm_files = os.listdir(osm_dir)
    for of in tqdm(osm_files):
        if of != "New-York-4km.osm":
            continue
        get_osm_images(os.path.join(osm_dir, of), zoom_level)
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
