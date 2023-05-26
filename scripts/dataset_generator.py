# -*- coding: utf-8 -*-
#
# @File:   dataset_generator.py
# @Author: Haozhe Xie
# @Date:   2023-03-31 15:04:25
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-05-26 10:49:47
# @Email:  root@haozhexie.com

import argparse
import cv2
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import torch

from tqdm import tqdm
from PIL import Image

# Disable the warning message for PIL decompression bomb
# Ref: https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import utils.helpers
import utils.osm_helper
import extensions.voxlib as voxlib
from extensions.extrude_tensor import TensorExtruder


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
        # Ignore underground highways
        return 0 if "layer" in highway_tags and highway_tags["layer"] < 0 else 1
        # return 1
    else:
        raise Exception("Unknown map name: %s" % map_name)


def _get_footprint_color(map_name, footprint_tags):
    if map_name == "height_field":
        if _tag_equals(footprint_tags, "height"):
            return int(float(footprint_tags["height"]) + 0.5)
        elif _tag_equals(footprint_tags, "role", ["inner"]):
            # "role" in footprint_tags and footprint_tags["role"] == "inner"
            return None
        elif _tag_equals(footprint_tags, "building", ["roof"]):
            #  "building" in footprint_tags and footprint_tags["building"] == "roof"
            return None
        elif _tag_equals(footprint_tags, "landuse", ["construction"]):
            #  "building" in footprint_tags and footprint_tags["building"] == "construction"
            return 10
        else:
            raise Exception("Unknown height for tag: %s" % footprint_tags)
    elif map_name == "seg_map":
        if _tag_equals(footprint_tags, "role", ["inner"]):
            return 2
        elif _tag_equals(footprint_tags, "building") or _tag_equals(
            footprint_tags, "building:part"
        ):
            return 2
        elif _tag_equals(footprint_tags, "landuse", ["construction"]):
            return 4
        else:
            return 0
    elif map_name == "footprint_contour":
        if _tag_equals(footprint_tags, "building"):
            return 1
        else:
            return 0
    else:
        raise Exception("Unknown map name: %s" % map_name)


def _remove_mask_outliers(mask):
    N_PIXELS_THRES = 96
    mask = cv2.erode(mask, np.ones((5, 5), dtype=np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    ignored_indexes = np.where(stats[:, -1] <= N_PIXELS_THRES)[0]
    ignored_mask = np.isin(labels, ignored_indexes)
    return mask * ~ignored_mask


def get_coast_zones(osm_tile_img_path, img_size):
    tile_img = cv2.imread(osm_tile_img_path)
    water_lb = np.array([219, 219, 195], dtype=np.uint8)
    water_ub = np.array([235, 235, 219], dtype=np.uint8)
    mask = cv2.resize(cv2.inRange(tile_img, water_lb, water_ub), img_size[::-1])
    return _remove_mask_outliers(mask)


def get_green_lands(osm_tile_img_path, seg_map):
    tile_img = cv2.imread(osm_tile_img_path)
    green_lb = np.array([209, 217, 217], dtype=np.uint8)
    green_ub = np.array([219, 236, 233], dtype=np.uint8)
    # Only assign green lands to uncategoried pixels
    uncategoried = np.zeros_like(seg_map)
    uncategoried[seg_map == 0] = True
    mask = cv2.resize(cv2.inRange(tile_img, green_lb, green_ub), seg_map.shape[::-1])
    return _remove_mask_outliers(mask) * uncategoried


def get_osm_images(osm_file_path, osm_tile_img_path, zoom_level):
    logging.debug("Reading OSM file[Path=%s] ..." % osm_file_path)
    highways, footprints, nodes = utils.osm_helper.get_highways_and_footprints(
        osm_file_path
    )
    resolution = utils.osm_helper.get_map_resolution(
        {
            k: float(v)
            for k, v in utils.osm_helper.get_lnglat_bounds(osm_file_path).items()
        },
        zoom_level,
    )
    nodes = utils.osm_helper.get_nodes_xy_coordinates(nodes, resolution, zoom_level)
    xy_bounds = utils.osm_helper.get_xy_bounds(nodes)

    # Fix missing height (for buildings) and width for highways
    highways = utils.osm_helper.fix_missing_highway_width(highways)
    footprints = utils.osm_helper.fix_missing_footprint_height(
        footprints, utils.osm_helper.get_footprint_height_stat(footprints)
    )

    # Generate semantic labels
    logging.debug("Generating segmentation maps ...")
    seg_map = utils.osm_helper.get_empty_map(xy_bounds)
    seg_map = utils.osm_helper.plot_highways(
        "seg_map", _get_highway_color, seg_map, highways, nodes, xy_bounds, resolution
    )
    # Read coast zones from no-label tile images
    logging.debug("Reading green lands and coast zones from tile images ...")
    coast_zones = None
    green_lands = None
    if not os.path.exists(osm_tile_img_path):
        logging.warning(
            "The coast zones for the OSM[Path=%s] could not be parsed due to a missing tile image[Path=%s]"
            % (osm_file_path, osm_tile_img_path)
        )
    else:
        green_lands = get_green_lands(osm_tile_img_path, seg_map)
        seg_map[green_lands != 0] = 3
        coast_zones = get_coast_zones(osm_tile_img_path, seg_map.shape)
        seg_map[coast_zones != 0] = 5
    # Plot footprint at the end to make building masks more complete
    seg_map = utils.osm_helper.plot_footprints(
        "seg_map",
        _get_footprint_color,
        seg_map,
        footprints,
        nodes,
        xy_bounds,
    )
    # Assign ID=6 to unlabelled pixels (regarded as ground)
    seg_map[seg_map == 0] = 6

    # Generate the contours of footprints
    logging.debug("Generating footprint contours ...")
    footprint_contour = utils.osm_helper.get_empty_map(xy_bounds)
    footprint_contour = utils.osm_helper.plot_footprints(
        "footprint_contour",
        _get_footprint_color,
        footprint_contour,
        footprints,
        nodes,
        xy_bounds,
        resolution,
    )
    footprint_contour = footprint_contour.astype(bool)

    # Generate height fields
    logging.debug("Generating height fields ...")
    height_field = utils.osm_helper.get_empty_map(xy_bounds, dtype=np.uint16)
    # highway_color == 0. Therefore, the following statement can be skipped.
    # height_field = utils.osm_helper.plot_highways(
    #     "height_field",
    #     _get_highway_color,
    #     height_field,
    #     highways,
    #     nodes,
    #     xy_bounds,
    #     resolution,
    # )
    height_field[height_field == 0] = 4
    if coast_zones is not None:
        height_field[coast_zones != 0] = 0
    if green_lands is not None:
        height_field[green_lands != 0] = 8
    # Follow the order in plotting seg maps
    height_field = utils.osm_helper.plot_footprints(
        "height_field",
        _get_footprint_color,
        height_field,
        footprints,
        nodes,
        xy_bounds,
    )

    # The height values should be normalized using the same scale as that of the width and height
    # dimensions. However, the following statement results in incorrect outputs. Quite STRANGE!
    # height_field = (height_field / resolution).astype(np.uint16)
    height_field = height_field.astype(np.uint16)
    return (
        height_field,
        seg_map,
        footprint_contour,
        {"resolution": resolution, "bounds": xy_bounds},
    )


def get_google_earth_projects(osm_basename, google_earth_dir):
    ge_projects = os.listdir(google_earth_dir)
    osm_info = osm_basename.split("-")
    osm_country, osm_city = osm_info[0], osm_info[1]
    projects = []
    for gp in ge_projects:
        if gp.startswith("%s-%s" % (osm_country, osm_city)):
            projects.append(gp)

    return sorted(projects)


def _get_google_earth_camera_poses(ge_proj_name, ge_dir):
    ge_proj_dir = os.path.join(ge_dir, ge_proj_name)
    camera_setting_file = os.path.join(ge_proj_dir, "%s.json" % ge_proj_name)
    ge_project_file = os.path.join(ge_proj_dir, "%s.esp" % ge_proj_name)
    if not os.path.exists(camera_setting_file) or not os.path.exists(ge_project_file):
        return None

    camera_settings = None
    with open(camera_setting_file) as f:
        camera_settings = json.load(f)

    camera_target = None
    with open(ge_project_file) as f:
        ge_proj_settings = json.load(f)
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
        "elevation": 0,
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
                "altitude": next(
                    _attr["value"]["relative"]
                    for _attr in camera_target
                    if _attr["type"] == "altitudePOI"
                )
                + 1,
            }
        },
        "poses": [],
    }
    # NOTE: The conversion from latitudePOI to latitude is unclear.
    #       Fixed with the collected metadata.
    extra_metadata_file = os.path.join(ge_proj_dir, "metadata.json")
    if not os.path.exists(extra_metadata_file):
        logging.error(
            "The project %s is missing the extra metadata file, "
            "which could result in a misalignment between the footage and segmentation maps."
            % ge_proj_name
        )
    else:
        with open(extra_metadata_file) as f:
            extra_metadata = json.load(f)

        camera_poses["elevation"] = extra_metadata["elevation"]
        camera_poses["center"]["coordinate"]["latitude"] = extra_metadata["clat"]

    # NOTE: All Google Earth renderings are centered around an altitude of 1.
    if camera_poses["center"]["coordinate"]["altitude"] != 1:
        logging.warning("The altitude of the camera center is not 1.")
        return None

    for cf in camera_settings["cameraFrames"]:
        camera_poses["poses"].append({"coordinate": cf["coordinate"]})
    return camera_poses


def _get_img_patch(img, cx, cy, patch_size):
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
    return img[h_s:h_e, x_s:x_e]


def _get_instance_seg_map(part_seg_map, part_contours, patch_size, use_contours=False):
    BULIDING_MASK_ID = 2
    BLD_INS_LABEL_MIN = 10
    N_PIXELS_THRES = 16
    if use_contours:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            (1 - part_contours).astype(np.uint8), connectivity=4
        )
    else:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            (part_seg_map == BULIDING_MASK_ID).astype(np.uint8), connectivity=4
        )

    # Remove non-building instance masks
    labels[part_seg_map != BULIDING_MASK_ID] = 0
    # Remove too small buildings
    ignored_indexes = np.where(stats[:, -1] <= N_PIXELS_THRES)[0]
    # Set the label of small buildings to 6 (others)
    labels[np.isin(labels, ignored_indexes)] = 6
    # Building instance mask
    building_mask = labels != 0

    # Building Instance Mask starts from 10 (labels + 10)
    part_seg_map[part_seg_map == BULIDING_MASK_ID] = 0
    part_seg_map = (
        part_seg_map * (1 - building_mask)
        + (labels + BLD_INS_LABEL_MIN) * building_mask
    )
    assert np.max(labels) < 2147483648
    # NOTE: assert stats.shape[1] == 5, represents x, y, w, h, area of the components.
    # Convert x and y to dx and dy, where dx and dy denote the offsets to the center.
    stats = stats.astype(np.float32)
    stats[:, 0] -= patch_size // 2 + stats[:, 2] / 2
    stats[:, 1] -= patch_size // 2 + stats[:, 3] / 2
    return part_seg_map.astype(np.int32), stats[:, :4]


def _get_diffuse_shading_img(seg_map, depth2, raydirs, cam_ori_t):
    mc_rgb = np.array(seg_map.convert("RGB"))
    # Diffused shading, co-located light.
    first_intersection_depth = depth2[0, :, :, 0, None, :]
    first_intersection_point = (
        raydirs * first_intersection_depth + cam_ori_t[None, None, None, :]
    )
    fip_local_coords = torch.remainder(first_intersection_point, 1.0)
    fip_wall_proximity = torch.minimum(fip_local_coords, 1.0 - fip_local_coords)
    fip_wall_orientation = torch.argmin(fip_wall_proximity, dim=-1, keepdim=False)
    # 0: [1,0,0]; 1: [0,1,0]; 2: [0,0,1]
    lut = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=torch.float32,
        device=fip_wall_orientation.device,
    )
    fip_normal = lut[fip_wall_orientation]
    diffuse_shade = torch.abs(torch.sum(fip_normal * raydirs, dim=-1))

    mc_rgb = mc_rgb.astype(float) / 255
    mc_rgb = mc_rgb * diffuse_shade.cpu().numpy()
    mc_rgb = (mc_rgb ** (1 / 2.2)) * 255
    return Image.fromarray(mc_rgb.astype(np.uint8))


def get_google_earth_aligned_seg_maps(
    ge_project_name,
    google_earth_dir,
    height_field,
    contours,
    seg_map,
    patch_size,
    metadata,
    zoom_level,
    tensor_extruder,
    debug,
):
    logging.info("Parsing Google Earth Project: %s" % ge_project_name)
    ge_camera_poses = _get_google_earth_camera_poses(ge_project_name, google_earth_dir)
    if ge_camera_poses is None:
        return []

    assert ge_camera_poses is not None
    ge_camera_focal = (
        ge_camera_poses["height"] / 2 / np.tan(np.deg2rad(ge_camera_poses["vfov"]))
    )
    # Build semantic 3D volume
    logging.debug("Camera Target Center: %s" % ge_camera_poses["center"]["coordinate"])
    cx, cy = utils.osm_helper.lnglat2xy(
        ge_camera_poses["center"]["coordinate"]["longitude"],
        ge_camera_poses["center"]["coordinate"]["latitude"],
        metadata["resolution"],
        zoom_level,
        dtype=float,
    )
    cx -= metadata["bounds"]["xmin"]
    cy -= metadata["bounds"]["ymin"]
    tr_cx, tr_cy = int(cx + 0.5), int(cy + 0.5)
    logging.debug(
        "Map Information: Center=%s; Origin=%s; Size(HxW): %s"
        % (
            (cx, cy),
            (tr_cx, tr_cy),
            (
                metadata["bounds"]["ymax"] - metadata["bounds"]["ymin"],
                metadata["bounds"]["xmax"] - metadata["bounds"]["xmin"],
            ),
        )
    )
    part_hf = _get_img_patch(
        height_field,
        tr_cx,
        tr_cy,
        patch_size,
    ).astype(np.int32)
    # Consider the elevation of the local area
    part_hf += ge_camera_poses["elevation"]

    part_contours = _get_img_patch(
        contours,
        tr_cx,
        tr_cy,
        patch_size,
    )
    part_seg_map = _get_img_patch(
        seg_map,
        tr_cx,
        tr_cy,
        patch_size,
    ).astype(np.int32)
    part_seg_map, bld_bboxes = _get_instance_seg_map(
        part_seg_map, part_contours, patch_size
    )

    # Build 3D Semantic Volume
    seg_volume = tensor_extruder(
        torch.from_numpy(part_seg_map[None, None, ...]).cuda(),
        torch.from_numpy(part_hf[None, None, ...]).cuda(),
    ).squeeze()
    logging.debug("The shape of SegVolume: %s" % (seg_volume.size(),))
    # Convert camera position to the voxel coordinate system
    vol_cx, vol_cy = ((patch_size - 1) // 2, (patch_size - 1) // 2)

    seg_maps = []
    for gcp in tqdm(ge_camera_poses["poses"], desc="Project: %s" % ge_project_name):
        x, y = utils.osm_helper.lnglat2xy(
            gcp["coordinate"]["longitude"],
            gcp["coordinate"]["latitude"],
            metadata["resolution"],
            zoom_level,
            dtype=float,
        )
        gcp["position"] = {
            "x": x - metadata["bounds"]["xmin"],
            "y": y - metadata["bounds"]["ymin"],
            "z": gcp["coordinate"]["altitude"],
        }
        # Run ray-voxel intersection
        r"""Ray-voxel intersection CUDA kernel.
        Note: voxel_id = 0 and depth2 = NaN if there is no intersection along the ray
        Args:
            voxel_t (H x W x D tensor, int32): Full 3D voxel of MC block IDs.
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
        cam_ori_t = torch.tensor(
            [
                gcp["position"]["y"] - tr_cy + vol_cy,
                gcp["position"]["x"] - tr_cx + vol_cx,
                gcp["position"]["z"],
            ],
            dtype=torch.float32,
            device=seg_volume.device,
        )
        voxel_id, depth2, raydirs = voxlib.ray_voxel_intersection_perspective(
            seg_volume,
            cam_ori_t,
            torch.tensor(
                [
                    cy - gcp["position"]["y"],
                    cx - gcp["position"]["x"],
                    -gcp["position"]["z"],
                ],
                dtype=torch.float32,
                device=seg_volume.device,
            ),
            torch.tensor([0, 0, 1], dtype=torch.float32),
            # The MAGIC NUMBER to make it aligned with Google Earth Renderings
            ge_camera_focal * 2.06,
            [
                (ge_camera_poses["height"] - 1) / 2.0,
                (ge_camera_poses["width"] - 1) / 2.0,
            ],
            [ge_camera_poses["height"], ge_camera_poses["width"]],
            N_MAX_SAMPLES,
        )
        # print(voxel_id.size())    # torch.Size([540, 960, 10, 1])
        if debug:
            seg_map = utils.helpers.get_seg_map(
                voxel_id.squeeze()[..., 0].cpu().numpy()
            )
            seg_maps.append(
                _get_diffuse_shading_img(seg_map, depth2, raydirs, cam_ori_t)
            )
        else:
            seg_maps.append(
                {
                    "voxel_id": voxel_id.cpu().numpy(),
                    "depth2": depth2.permute(1, 2, 0, 3, 4).cpu().numpy(),
                    "raydirs": raydirs.cpu().numpy(),
                    "cam_ori_t": cam_ori_t.cpu().numpy(),
                    "img_center": {"cx": tr_cx, "cy": tr_cy},
                }
            )

    return seg_maps, bld_bboxes


def get_ambiguous_seg_mask(voxel_id, est_seg_map):
    BULIDING_MASK_ID = 2
    BLD_INS_LABEL_MIN = 10
    seg_map = voxel_id.squeeze()[..., 0]
    seg_map[seg_map >= BLD_INS_LABEL_MIN] = BULIDING_MASK_ID
    est_seg_map = np.array(est_seg_map.convert("P"))
    return seg_map == est_seg_map


def main(
    osm_dir,
    ges_dir,
    seg_dir,
    osm_out_dir,
    ges_out_dir,
    patch_size,
    max_height,
    zoom_level,
    debug,
):
    osm_files = sorted([f for f in os.listdir(osm_dir) if f.endswith(".osm")])
    tensor_extruder = TensorExtruder(max_height)
    for of in tqdm(osm_files):
        basename, _ = os.path.splitext(of)
        # Create folder for the OSM
        _osm_out_dir = os.path.join(osm_out_dir, basename)
        os.makedirs(_osm_out_dir, exist_ok=True)
        # Rasterisation
        output_hf_file_path = os.path.join(_osm_out_dir, "hf.png")
        output_ctr_file_path = os.path.join(_osm_out_dir, "ctr.png")
        output_seg_map_file_path = os.path.join(_osm_out_dir, "seg.png")
        metadata_file_path = os.path.join(_osm_out_dir, "metadata.json")
        if (
            os.path.exists(output_hf_file_path)
            and os.path.exists(output_ctr_file_path)
            and os.path.exists(output_seg_map_file_path)
            and os.path.exists(metadata_file_path)
        ):
            height_field = np.array(Image.open(output_hf_file_path))
            contours = np.array(Image.open(output_ctr_file_path))
            seg_map = np.array(Image.open(output_seg_map_file_path).convert("P"))
            with open(metadata_file_path) as f:
                metadata = json.load(f)
        else:
            height_field, seg_map, contours, metadata = get_osm_images(
                os.path.join(osm_dir, of),
                os.path.join(_osm_out_dir, "tiles.png"),
                zoom_level,
            )
            Image.fromarray(height_field).save(output_hf_file_path)
            Image.fromarray(contours).save(output_ctr_file_path)
            utils.helpers.get_seg_map(seg_map).save(output_seg_map_file_path)
            with open(metadata_file_path, "w") as f:
                json.dump(metadata, f)

        # Align images from Google Earth Studio
        logging.debug("Generating Google Earth segmentation maps ...")
        ge_projects = get_google_earth_projects(basename, ges_dir)
        if not ge_projects:
            logging.warning(
                "No matching Google Earth Project found for OSM[File=%s]." % of
            )
            continue
        # Read Google Earth Studio metadata
        for gep in ge_projects:
            seg_maps, bld_offsets = get_google_earth_aligned_seg_maps(
                gep,
                ges_dir,
                height_field,
                contours,
                seg_map,
                patch_size,
                metadata,
                zoom_level,
                tensor_extruder,
                debug,
            )
            # Generate the corresponding voxel raycasting maps
            _ges_out_dir = ges_out_dir % gep
            os.makedirs(_ges_out_dir, exist_ok=True)
            np.save(os.path.join(_ges_out_dir, os.pardir, "%s.npy" % gep), bld_offsets)
            for idx, sg in enumerate(seg_maps):
                if debug:
                    sg.save(os.path.join(_ges_out_dir, "%s-%02d.jpg" % (gep, idx)))
                else:
                    with open(
                        os.path.join(_ges_out_dir, "%s_%02d.pkl" % (gep, idx)), "wb"
                    ) as f:
                        sg["mask"] = get_ambiguous_seg_mask(
                            sg["voxel_id"].copy(),
                            Image.open(
                                os.path.join(seg_dir % gep, "%s_%02d.png" % (gep, idx))
                            ),
                        )
                        pickle.dump(sg, f)


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (36, 36)
    logging.basicConfig(
        filename=os.path.join(PROJECT_HOME, "output", "dataset-generator.log"),
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.DEBUG,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--osm_dir", default=os.path.join(PROJECT_HOME, "data", "xml"))
    parser.add_argument("--ges_dir", default=os.path.join(PROJECT_HOME, "data", "ges"))
    parser.add_argument(
        "--seg_dir", default=os.path.join(PROJECT_HOME, "data", "ges", "%s", "seg")
    )
    parser.add_argument(
        "--osm_out_dir", default=os.path.join(PROJECT_HOME, "data", "osm")
    )
    parser.add_argument(
        "--ges_out_dir",
        default=os.path.join(PROJECT_HOME, "data", "ges", "%s", "raycasting"),
    )
    parser.add_argument("--patch_size", default=1536)
    parser.add_argument("--max_height", default=640)
    parser.add_argument("--zoom", default=18)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(
        args.osm_dir,
        args.ges_dir,
        args.seg_dir,
        args.osm_out_dir,
        args.ges_out_dir,
        args.patch_size,
        args.max_height,
        args.zoom,
        args.debug,
    )
