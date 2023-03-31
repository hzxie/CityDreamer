# -*- coding: utf-8 -*-
#
# @File:   dataset_generator.py
# @Author: Haozhe Xie
# @Date:   2023-03-21 18:26:26
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-03-31 16:35:10
# @Email:  root@haozhexie.com

import argparse
import cv2
import json
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch

from tqdm import tqdm
from PIL import Image

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import utils.osm_helper
import extensions.voxlib as voxlib
from extensions.cu_extrude_tensor import TensorExtruder


def get_highways_and_footprints(osm_file_path):
    highways, nodes = utils.osm_helper.get_highways(
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
    footprints, nodes = utils.osm_helper.get_footprints(
        osm_file_path,
        [
            "building",
            {"k": "landuse", "v": ["construction"]},
            {"k": "leisure", "v": ["park", "marina"]},
        ],
        nodes,
    )
    coastlines, nodes = utils.osm_helper.get_highways(
        osm_file_path, [{"k": "natural", "v": "coastline"}], nodes
    )
    nodes = utils.osm_helper.get_nodes_lng_lat(osm_file_path, nodes)
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
        k: float(v)
        for k, v in utils.osm_helper.get_lnglat_bounds(osm_file_path).items()
    }
    resolution = utils.osm_helper.get_map_resolution(lnglat_bounds, zoom_level)
    nodes = utils.osm_helper.get_nodes_xy_coordinates(nodes, resolution, zoom_level)
    xy_bounds = utils.osm_helper.get_xy_bounds(nodes)

    # Fix missing height (for buildings) and width (for highways and coastlines)
    highways = utils.osm_helper.fix_missing_highway_width(highways)
    coastlines = utils.osm_helper.fix_missing_highway_width(coastlines)
    footprints = utils.osm_helper.fix_missing_footprint_height(
        footprints, utils.osm_helper.get_footprint_height_stat(footprints)
    )

    # Generate semantic labels
    seg_map = utils.osm_helper.get_empty_map(xy_bounds)
    seg_map = utils.osm_helper.plot_highways(
        "seg_map", _get_highway_color, seg_map, highways, nodes, xy_bounds, resolution
    )
    seg_map = utils.osm_helper.plot_highways(
        "seg_map", _get_highway_color, seg_map, coastlines, nodes, xy_bounds, resolution
    )
    seg_map = utils.osm_helper.plot_footprints(
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
    height_field = utils.osm_helper.get_empty_map(xy_bounds)
    height_field = utils.osm_helper.plot_footprints(
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

    return height_field, seg_map, {"resolution": resolution, "bounds": xy_bounds}


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


def get_google_earth_project_name(osm_basename, google_earth_dir):
    ge_projects = os.listdir(google_earth_dir)
    osm_info = osm_basename.split("-")
    osm_country, osm_city = osm_info[0], osm_info[1]
    for gp in ge_projects:
        if gp.startswith("%s-%s" % (osm_country, osm_city)):
            return gp
    return None


def get_google_earth_camera_poses(ge_proj_name, ge_dir):
    ge_proj_dir = os.path.join(ge_dir, ge_proj_name)
    camera_setting_file = os.path.join(ge_proj_dir, "%s.json" % ge_proj_name)
    ge_project_file = os.path.join(ge_proj_dir, "%s.esp" % ge_proj_name)
    if not os.path.exists(camera_setting_file) or not os.path.exists(ge_project_file):
        return None

    camera_settings = None
    with open(camera_setting_file) as f:
        camera_settings = json.loads(f.read())

    camera_target = None
    with open(ge_project_file) as f:
        ge_proj_settings = json.loads(f.read())
        scene = ge_proj_settings["scenes"][0]["attributes"]
        camera_group = next(
            _attr["attributes"] for _attr in scene if _attr["type"] == "cameraGroup"
        )
        camera_taget_effect = next(
            _attr["attributes"]
            for _attr in camera_group
            if _attr["type"] == "cameraTargetEffect"
        )
        camera_target = next(
            _attr["attributes"]
            for _attr in camera_taget_effect
            if _attr["type"] == "poi"
        )

    camera_poses = {
        "vfov": camera_settings["cameraFrames"][0]["fovVertical"],
        "width": camera_settings["width"],
        "height": camera_settings["height"],
        "center": {
            "coordinate": {
                "longitude": next(
                    _attr["value"]["relative"]
                    for _attr in camera_target
                    if _attr["type"] == "longitudePOI"
                )
                * 360
                - 180,
                "latitude": next(
                    _attr["value"]["relative"]
                    for _attr in camera_target
                    if _attr["type"] == "latitudePOI"
                )
                * 180
                - 90,
            }
        },
        "poses": [],
    }
    for cf in camera_settings["cameraFrames"]:
        camera_poses["poses"].append(
            {"rotation": cf["rotation"], "coordinate": cf["coordinate"]}
        )
    return camera_poses


def get_rotation_matrix_local_cord(center_lng, center_lat):
    return np.array(
        [
            [-math.sin(center_lng), math.cos(center_lng), 0],
            [
                -math.sin(center_lat) * math.cos(center_lng),
                -math.sin(center_lat) * math.sin(center_lng),
                math.cos(center_lat),
            ],
            [
                math.cos(center_lat) * math.cos(center_lng),
                math.cos(center_lat) * math.sin(center_lng),
                math.sin(center_lat),
            ],
        ]
    )


def _is_rotation_matrix(r):
    r_t = np.transpose(r)
    should_be_identity = np.dot(r_t, r)
    I = np.identity(3, dtype=r.dtype)
    n = np.linalg.norm(I - should_be_identity)
    return n < 1e-6


def _euler_angles_to_rotation_matrix(theta):
    rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )
    ry = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )
    rz = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    return np.dot(rz, np.dot(ry, rx))


def _rotation_matrix_to_euler_angles(r):
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0

    return (x, y, z)


def get_euler_angles_local_cord(theta_wc, rot_mtx_lc):
    # Ref:
    # - http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    # - https://learnopencv.com/rotation-matrix-to-euler-angles/
    cm_r_wc = _euler_angles_to_rotation_matrix(theta_wc)
    cm_rot_mat = np.dot(np.linalg.inv(cm_r_wc), rot_mtx_lc)
    assert _is_rotation_matrix(cm_rot_mat)
    return _rotation_matrix_to_euler_angles(cm_rot_mat)


def _get_img_patch_tensor(img, cx, cy, patch_size):
    h, w = img.shape
    x_s, x_e = cx - patch_size // 2, cx + patch_size // 2
    h_s, h_e = cy - patch_size // 2, cy + patch_size // 2
    if x_s < 0 or x_e >= w:
        x_s = 0 if x_s < 0 else x_s
        x_e = 0 if x_e >= w else x_e
        logging.error("The horizontal center is not located at %d as expected" % cx)
    if h_s < 0 or h_e >= h:
        h_s = 0 if h_s < 0 else h_s
        h_e = 0 if h_e >= h else h_e
        logging.error("The vertical center is not located at %d as expected" % cy)
    return (
        torch.from_numpy(img[h_s:h_e, x_s:x_e].astype(np.int32))
        .unsqueeze(dim=0)
        .unsqueeze(dim=0)
        .cuda()
    )


def main(osm_dir, google_earth_dir, patch_size, max_height, zoom_level):
    osm_files = [f for f in os.listdir(osm_dir) if f.endswith(".osm")]
    tensor_extruder = TensorExtruder(max_height)
    for of in tqdm(osm_files):
        basename, _ = os.path.splitext(of)
        # Create folder for the OSM
        _osm_dir = os.path.join(osm_dir, basename)
        os.makedirs(_osm_dir, exist_ok=True)
        # Rasterisation
        height_field, seg_map, metadata = get_osm_images(
            os.path.join(osm_dir, of), zoom_level
        )
        Image.fromarray(height_field).save(os.path.join(_osm_dir, "hf.png"))
        get_seg_map_img(seg_map).save(os.path.join(_osm_dir, "seg.png"))
        # Align images from Google Earth Studio
        _ge_proj_name = get_google_earth_project_name(basename, google_earth_dir)
        if _ge_proj_name is None:
            logging.warning(
                "No matching Google Earth Project found for OSM[File=%s]." % of
            )
            continue
        ## Read Google Earth Studio metadata
        ge_camera_poses = get_google_earth_camera_poses(_ge_proj_name, google_earth_dir)
        ge_camera_focal = (
            ge_camera_poses["height"] / 2 / np.tan(np.deg2rad(ge_camera_poses["vfov"]))
        )
        ## Build semantic 3D volume
        cx, cy = utils.osm_helper.lnglat2xy(
            ge_camera_poses["center"]["coordinate"]["longitude"],
            ge_camera_poses["center"]["coordinate"]["latitude"],
            metadata["resolution"],
            zoom_level,
        )
        ge_camera_poses["center"]["position"] = {
            "x": cx - metadata["bounds"]["xmin"],
            "y": cy - metadata["bounds"]["ymin"],
        }
        logging.debug(
            "Map Information: Center=%s; Size(HxW): %s"
            % (
                ge_camera_poses["center"]["position"],
                (
                    metadata["bounds"]["ymax"] - metadata["bounds"]["ymin"],
                    metadata["bounds"]["xmax"] - metadata["bounds"]["xmin"],
                ),
            )
        )
        seg_volume = (
            tensor_extruder(
                _get_img_patch_tensor(
                    seg_map,
                    ge_camera_poses["center"]["position"]["x"],
                    ge_camera_poses["center"]["position"]["y"],
                    patch_size,
                ),
                _get_img_patch_tensor(
                    height_field,
                    ge_camera_poses["center"]["position"]["x"],
                    ge_camera_poses["center"]["position"]["y"],
                    patch_size,
                ),
            )
            .squeeze()
            .permute(2, 0, 1)
        )
        logging.debug("The shape of SegVolume: %s" % (seg_volume.size(),))
        ## Convert camera poses to the local coordinate system
        rot_mtx_cvt_lc = get_rotation_matrix_local_cord(
            ge_camera_poses["center"]["coordinate"]["longitude"],
            ge_camera_poses["center"]["coordinate"]["latitude"],
        )
        for idx, gcp in enumerate(
            tqdm(ge_camera_poses["poses"], desc="Project: %s" % _ge_proj_name)
        ):
            x, y = utils.osm_helper.lnglat2xy(
                gcp["coordinate"]["longitude"],
                gcp["coordinate"]["latitude"],
                metadata["resolution"],
                zoom_level,
            )
            gcp["position"] = {
                "x": x - metadata["bounds"]["xmin"],
                "y": y - metadata["bounds"]["ymin"],
                "z": gcp["coordinate"]["altitude"],
            }
            # Ref: https://github.com/city-super/BungeeNeRF/blob/26cc7f4c848e20961dbf067114faa4268034349e/GES2pose.py#L94-L96
            _x = math.radians(-gcp["rotation"]["x"])
            _y = math.radians(180 - gcp["rotation"]["y"])
            _z = math.radians(180 + gcp["rotation"]["z"])
            cm_theta_lc = get_euler_angles_local_cord((_x, _y, _z), rot_mtx_cvt_lc)
            gcp["rotation"] = {
                "x": cm_theta_lc[0],
                "y": cm_theta_lc[1],
                "z": cm_theta_lc[2],
            }
            ## Run ray-voxel intersection
            r"""Ray-voxel intersection CUDA kernel.
            Note: voxel_id = 0 and depth2 = NaN if there is no intersection along the ray
            Args:
                voxel_t (height x size x size tensor, int32): Full 3D voxel of MC block IDs.
                cam_ori_t (3 tensor): Camera origin.
                cam_dir_t (3 tensor): Camera direction.
                cam_up_t (3 tensor): Camera up vector.
                cam_f (float): Camera focal length (in pixels).
                cam_c  (list of 2 floats [x, y]): Camera optical center.
                img_dims (list of 2 ints [H, W]): Camera resolution.
                max_samples (int): Maximum number of blocks intersected along the ray before stopping.
            Returns:
                voxel_id (    img_dims[0] x img_dims[1] x max_samples x 1 tensor): IDs of intersected tensors
                along each ray
                depth2   (2 x img_dims[0] x img_dims[1] x max_samples x 1 tensor): Depths of entrance and exit
                points for each ray-voxel intersection.
                raydirs  (    img_dims[0] x img_dims[1] x 1 x 3 tensor): The direction of each ray.

            """
            N_MAX_SAMPLES = 6
            voxel_id, depth2, raydirs = voxlib.ray_voxel_intersection_perspective(
                seg_volume,
                torch.tensor(
                    [gcp["position"]["x"], gcp["position"]["y"], gcp["position"]["z"]],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [gcp["rotation"]["x"], gcp["rotation"]["y"], gcp["rotation"]["z"]],
                    dtype=torch.float32,
                ),
                torch.tensor([1, 0, 0], dtype=torch.float32),
                ge_camera_focal,
                [
                    (ge_camera_poses["width"] - 1) / 2.0,
                    (ge_camera_poses["height"] - 1) / 2.0,
                ],
                [ge_camera_poses["height"], ge_camera_poses["width"]],
                N_MAX_SAMPLES,
            )
            # print(voxel_id.size())    # torch.Size([540, 960, 10, 1])
            ## Generate the corresponding segmentation images
            ges_seg_dir = os.path.join(google_earth_dir, _ge_proj_name, "seg")
            os.makedirs(ges_seg_dir, exist_ok=True)
            get_seg_map_img(voxel_id.squeeze()[..., 0].cpu().numpy()).save(
                os.path.join(ges_seg_dir, "%s-%04d.png" % (_ge_proj_name, idx))
            )
            # Debug
            print(torch.sum(torch.sum(voxel_id)))
            break


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (48, 30)
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--osm_dir", default=os.path.join(PROJECT_HOME, "data", "osm"))
    parser.add_argument("--ges_dir", default=os.path.join(PROJECT_HOME, "data", "ges"))
    parser.add_argument("--patch_size", default=1024)
    parser.add_argument("--max_height", default=256)
    parser.add_argument("--zoom", default=18)
    args = parser.parse_args()
    main(args.osm_dir, args.ges_dir, args.patch_size, args.max_height, args.zoom)
