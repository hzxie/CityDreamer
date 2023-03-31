# -*- coding: utf-8 -*-
#
# @File:   osm_helper.py
# @Author: Haozhe Xie
# @Date:   2023-03-21 16:16:06
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-03-31 13:52:04
# @Email:  root@haozhexie.com

import cv2
import logging
import lxml.etree
import math
import numpy as np


def get_lnglat_bounds(xml_file_path):
    root = lxml.etree.parse(xml_file_path)
    bounds = root.find("bounds")
    return {
        "minlat": bounds.get("minlat"),
        "maxlat": bounds.get("maxlat"),
        "minlon": bounds.get("minlon"),
        "maxlon": bounds.get("maxlon"),
    }


def get_xy_bounds(nodes):
    min_x, min_y = math.inf, math.inf
    max_x, max_y = 0, 0
    for _, values in nodes.items():
        if values["x"] < min_x:
            min_x = values["x"]
        if values["x"] > max_x:
            max_x = values["x"]
        if values["y"] < min_y:
            min_y = values["y"]
        if values["y"] > max_y:
            max_y = values["y"]

    return {
        "xmin": math.ceil(min_x),
        "xmax": math.floor(max_x),
        "ymin": math.ceil(min_y),
        "ymax": math.floor(max_y),
    }


def get_nodes_lng_lat(osm_file_path, nodes):
    root = lxml.etree.parse(osm_file_path)
    for node in root.findall("node"):
        node_id = node.get("id")
        if node_id not in nodes:
            continue
        if node.get("action") == "delete":
            continue
        if node.get("lon") is None or node.get("lat") is None:
            continue
        nodes[node_id]["lng"] = float(node.get("lon"))
        nodes[node_id]["lat"] = float(node.get("lat"))

    return nodes


def _is_interested_way(tags, interested_tags):
    for it in interested_tags:
        if type(it) == str:
            if it in tags:
                return True
        elif type(it) == dict:
            if it["k"] in tags and tags[it["k"]] in it["v"]:
                return True
    return False


def get_highways(osm_file_path, highway_tags=[], highway_nodes=None):
    highways = {}
    if highway_nodes is None:
        highway_nodes = {}

    root = lxml.etree.parse(osm_file_path)
    for way in root.findall("way"):
        way_id = way.get("id")
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        if not _is_interested_way(tags, highway_tags) or (
            "action" in tags and tags["action"] == "delete"
        ):
            continue

        _highway_nodes = [n.get("ref") for n in way.findall("nd")]
        if not _highway_nodes:
            continue

        _get_width = lambda w: float("".join(c for c in w if c.isdigit()))
        highways[way_id] = {
            "nodes": _highway_nodes,
            "tags": tags,
        }
        # highways[way_id]["layer"] = tags["layer"] if "layer" in tags else 0
        highways[way_id]["tags"]["width"] = (
            _get_width(tags["width"]) if "width" in tags else None
        )
        for hn in _highway_nodes:
            if hn not in highway_nodes:
                highway_nodes[hn] = {}
            highway_nodes[hn]["nid"] = hn

    return highways, highway_nodes


def _get_footprint_nodes(way, nodes):
    _footprint_nodes = [n.get("ref") for n in way.findall("nd")]
    if _footprint_nodes:
        for node_id in _footprint_nodes:
            if node_id not in nodes:
                nodes[node_id] = {"nid": node_id}

    return _footprint_nodes


def get_footprints(xml_file_path, footprint_tags=[], footprint_nodes=None):
    footprints = {}
    if footprint_nodes is None:
        footprint_nodes = {}

    root = lxml.etree.parse(xml_file_path)
    relational_ways = {}
    for way in root.findall("relation"):
        members = way.findall("member")
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        if not _is_interested_way(tags, footprint_tags) or (
            "action" in tags and tags["action"] == "delete"
        ):
            continue
        for member in members:
            way_id = member.get("ref")
            role = member.get("role")
            relational_ways[way_id] = tags
            relational_ways[way_id]["role"] = role

    for way in root.findall("way"):
        way_id = way.get("id")
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        if "action" in tags and tags["action"] == "delete":
            continue
        if _is_interested_way(tags, footprint_tags) or way_id in relational_ways:
            footprints[way_id] = {
                "nodes": _get_footprint_nodes(way, footprint_nodes),
                "tags": tags,
            }
            if "height" in tags:
                try:
                    tags["height"] = float(tags["height"])
                except:
                    logging.warning(
                        "Invalid height value %s for Footprint[ID=%s]"
                        % (tags["height"], way_id)
                    )
                    del tags["height"]

            if way_id in relational_ways:
                footprints[way_id]["tags"].update(relational_ways[way_id])

    return footprints, footprint_nodes


def get_map_resolution(lnglat_bounds, zoom_level):
    # Reference: https://stackoverflow.com/questions/44223387/how-much-longitude-and-latitude-does-a-pixel-represent-in-google-maps-with-zero
    return (
        156543.03
        * math.cos(
            math.radians((lnglat_bounds["minlat"] + lnglat_bounds["maxlat"]) / 2)
        )
        / (2.0**zoom_level)
    )


def lnglat2xy(lng, lat, resolution, zoom_level, tile_size=256):
    n = 2.0**zoom_level
    x = (lng + 180.0) / 360.0 * n * tile_size
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n * tile_size
    return (int(x * resolution + 0.5), int(y * resolution + 0.5))


def get_nodes_xy_coordinates(nodes, resolution, zoom_level):
    for _, values in nodes.items():
        if "lng" not in values or "lat" not in values:
            logging.warning("Missing values for Node[ID=%s]" % values["nid"])
            continue
        values["x"], values["y"] = lnglat2xy(
            values["lng"], values["lat"], resolution, zoom_level
        )
    return nodes


def get_empty_map(xy_bounds):
    map_img = np.zeros(
        (
            xy_bounds["ymax"] - xy_bounds["ymin"] + 1,
            xy_bounds["xmax"] - xy_bounds["xmin"] + 1,
        ),
        dtype=np.int16,
    )
    return map_img


def fix_missing_highway_width(highways):
    for _, values in highways.items():
        if "width" not in values["tags"] or values["tags"]["width"] is None:
            values["tags"]["width"] = _get_missing_highway_width(values["tags"])
    return highways


def _get_missing_highway_width(highway_tags):
    highway_level = highway_tags["highway"] if "highway" in highway_tags else None
    if highway_level in ["motorway", "trunk"]:
        return 4.65 * 7
    elif highway_level == "primary":
        return 4.65 * 5
    elif highway_level == "secondary":
        return 4.65 * 3
    # elif highway_level == "secondary_link":
    #     return 5
    elif highway_level in ["motorway_link", "trunk_link", "primary_link"]:
        return 4.65 * 2
    elif highway_level in ["tertiary", "service", "residential"]:
        return 4.65 * 2
    else:
        return 4.65


def get_footprint_height_stat(footprints):
    height = []
    for _, values in footprints.items():
        if "height" in values["tags"]:
            height.append(float(values["tags"]["height"]))
    return {"1/4": np.percentile(height, 25), "3/4": np.percentile(height, 75)}


def fix_missing_footprint_height(footprints, footprint_height_stat):
    for _, values in footprints.items():
        if "height" not in values["tags"] or values["tags"]["height"] is None:
            values["tags"]["height"] = _get_missing_footprint_height(
                values, footprint_height_stat
            )
    return footprints


def _get_missing_footprint_height(footprint_tags, footprint_height_stat):
    return int(
        np.random.uniform(footprint_height_stat["1/4"], footprint_height_stat["3/4"])
    )


def plot_highways(
    map_name, colormap, map_img, highways, highway_nodes, xy_bounds, resolution
):
    for _, values in highways.items():
        way_nodes = []
        for nid in values["nodes"]:
            node = highway_nodes[nid]
            way_nodes.append(
                ((node["x"] - xy_bounds["xmin"], node["y"] - xy_bounds["ymin"]))
            )
        if values["tags"]["width"] is None:
            continue

        cv2.polylines(
            map_img,
            [np.int32(way_nodes)],
            isClosed=False,
            color=colormap(map_name, values["tags"]),
            thickness=math.floor(values["tags"]["width"] / resolution + 0.5),
        )
    return map_img


def plot_footprints(
    map_name, colormap, map_img, footprints, footprint_nodes, xy_bounds
):
    for _, values in footprints.items():
        way_nodes = []
        for nid in values["nodes"]:
            node = footprint_nodes[nid]
            way_nodes.append(
                ((node["x"] - xy_bounds["xmin"], node["y"] - xy_bounds["ymin"]))
            )
        # color is None for ignored footprints
        color = colormap(map_name, values["tags"])
        if color is not None:
            cv2.fillPoly(
                map_img,
                [np.int32(way_nodes)],
                color=color,
            )

    return map_img
