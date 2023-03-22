# -*- coding: utf-8 -*-
#
# @File:   osm_helper.py
# @Author: Haozhe Xie
# @Date:   2023-03-21 16:16:06
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-03-22 20:50:53
# @Email:  root@haozhexie.com

import cv2
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


def get_highways(osm_file_path, highway_tags=[], highway_nodes=None):
    highways = {}
    if highway_nodes is None:
        highway_nodes = {}

    root = lxml.etree.parse(osm_file_path)
    for way in root.findall("way"):
        way_id = way.get("id")
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        if not tags or "highway" not in tags:
            continue
        if "action" in tags and tags["action"] == "delete":
            continue

        _highway_nodes = [n.get("ref") for n in way.findall("nd")]
        if not _highway_nodes:
            continue

        _get_width = lambda w: float("".join(c for c in w if c.isdigit()))
        highways[way_id] = {
            "level": tags["highway"] if "highway" in tags else None,
            "layer": tags["layer"] if "layer" in tags else 0,
            "width": _get_width(tags["width"]) if "width" in tags else None,
            "nodes": _highway_nodes,
        }

        for hn in _highway_nodes:
            if hn not in highway_nodes:
                highway_nodes[hn] = {"ways": []}
            highway_nodes[hn]["nid"] = hn
            highway_nodes[hn]["ways"].append(way_id)

    return highways, highway_nodes


def _get_footprint_nodes(way, nodes):
    _footprint_nodes = [n.get("ref") for n in way.findall("nd")]
    if _footprint_nodes:
        for node_id in _footprint_nodes:
            if node_id not in nodes:
                nodes[node_id] = {"nid": node_id}

    return _footprint_nodes


def _is_interested_footprint(tags, footprint_tags):
    for ft in footprint_tags:
        if type(ft) == str:
            if ft in tags:
                return True
        elif type(ft) == dict:
            if ft["k"] in tags and tags[ft["k"]] in ft["v"]:
                return True

    return False


def get_footprints(xml_file_path, footprint_tags=[], footprint_nodes=None):
    footprints = {}
    if footprint_nodes is None:
        footprint_nodes = {}

    root = lxml.etree.parse(xml_file_path)
    relational_ways = {}
    for way in root.findall("relation"):
        members = way.findall("member")
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        if not _is_interested_footprint(tags, footprint_tags) or (
            "action" in tags and tags["action"] == "delete"
        ):
            continue
        for member in members:
            way_id = member.get("ref")
            role = member.get("role")
            relational_ways[way_id] = {"role": role}

    for way in root.findall("way"):
        way_id = way.get("id")
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        if "action" in tags and tags["action"] == "delete":
            continue
        if _is_interested_footprint(tags, footprint_tags) or way_id in relational_ways:
            footprints[way_id] = {
                "nodes": _get_footprint_nodes(way, footprint_nodes),
            }
            footprints[way_id]["tags"] = tags
            if way_id in relational_ways:
                footprints[way_id]["tags"]["role"] = relational_ways[way_id]["role"]

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


def _lnglat2xy(lng, lat, resolution, zoom_level, tile_size=256):
    n = 2.0**zoom_level
    x = (lng + 180.0) / 360.0 * n * tile_size
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n * tile_size
    return (int(int(x) * resolution), int(int(y) * resolution))


def get_nodes_xy_coordinates(nodes, resolution, zoom_level):
    for _, values in nodes.items():
        if "lng" not in values or "lat" not in values:
            logging.warning("Missing values for Node[ID=%s]" % values["nid"])
            continue
        values["x"], values["y"] = _lnglat2xy(
            values["lng"], values["lat"], resolution, zoom_level
        )
    return nodes


def get_empty_map(xy_bounds):
    map_img = np.zeros(
        (
            xy_bounds["ymax"] - xy_bounds["ymin"] + 1,
            xy_bounds["xmax"] - xy_bounds["xmin"] + 1,
        ),
        dtype=np.uint8,
    )
    return map_img


def fix_missing_highway_width(highways):
    for _, values in highways.items():
        if "width" not in values or values["width"] is None:
            values["width"] = _get_missing_highway_width(values)
    return highways


def _get_missing_highway_width(highway_tags):
    highway_level = highway_tags["highway"] if "highway" in highway_tags else None
    if highway_level in ["motorway", "trunk"]:
        return 8 * 4.65
    elif highway_level == "primary":
        return 6 * 4.65
    elif highway_level == "secondary":
        return 4 * 4.65
    # elif highway_level == "secondary_link":
    #     return 4.65
    elif highway_level in ["motorway_link", "trunk_link", "primary_link"]:
        return 2 * 4.65
    elif highway_level in ["tertiary", "service", "residential"]:
        return 2 * 4.65
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
    if "role" in footprint_tags and footprint_tags["role"] == "inner":
        return 0
    elif "leisure" in footprint_tags and footprint_tags["leisure"] == "park":
        return 5
    elif "landuse" in footprint_tags and footprint_tags["landuse"] == "construction":
        return 10
    elif "building" in footprint_tags and footprint_tags["building"] == "roof":
        return 0
    elif "building:levels" in footprint_tags:
        return int(footprint_tags["building:levels"]) * 4.26

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
        if "width" not in values:
            print(_, values)
        cv2.polylines(
            map_img,
            [np.int32(way_nodes)],
            isClosed=False,
            color=colormap(map_name, values),
            thickness=math.floor(
                (
                    values["width"]
                    if values["width"] is not None
                    else _get_missing_highway_width(values)
                )
                / resolution
                + 0.5
            ),
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
        cv2.fillPoly(
            map_img,
            [np.int32(way_nodes)],
            color=colormap(map_name, values["tags"]),
        )

    return map_img
