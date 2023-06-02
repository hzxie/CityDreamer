# -*- coding: utf-8 -*-
#
# @File:   inference.py
# @Author: Haozhe Xie
# @Date:   2023-05-31 15:01:28
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-06-02 11:15:45
# @Email:  root@haozhexie.com

import argparse
import copy
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import sys

from PIL import Image
from tqdm import tqdm

# Disable the warning message for PIL decompression bomb
# Ref: https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import extensions.extrude_tensor
import extensions.voxlib
import models.vqgan
import models.sampler
import models.gancraft
import utils.helpers

CONSTANTS = {
    "BULIDING_MASK_ID": 2,
    "BLD_INS_LABEL_MIN": 10,
    "LAYOUT_MAX_HEIGHT": 640,
    "LAYOUT_N_CLASSES": 7,
    "LAYOUT_VOL_SIZE": 1536,
    "BUILDING_VOL_SIZE": 672,
    "GES_VFOV": 20,
    "GES_IMAGE_HEIGHT": 540,
    "GES_IMAGE_WIDTH": 960,
    "IMAGE_PADDING": 8,
    "N_VOXEL_INTERSECT_SAMPLES": 6,
}


def get_instance_seg_map(seg_map):
    # Copied from scripts/dataset_generator.py
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (seg_map == CONSTANTS["BULIDING_MASK_ID"]).astype(np.uint8), connectivity=4
    )
    # Remove non-building instance masks
    labels[seg_map != CONSTANTS["BULIDING_MASK_ID"]] = 0
    # Building instance mask
    building_mask = labels != 0

    # Building Instance Mask starts from 10 (labels + 10)
    seg_map[seg_map == CONSTANTS["BULIDING_MASK_ID"]] = 0
    seg_map = (
        seg_map * (1 - building_mask)
        + (labels + CONSTANTS["BLD_INS_LABEL_MIN"]) * building_mask
    )
    assert np.max(labels) < 2147483648
    return seg_map.astype(np.int32), stats[:, :4]


def get_city_layout(city_osm_dir=None, sampler=None, vqae=None, hf_seg=None):
    if city_osm_dir is not None:
        hf = np.array(Image.open(os.path.join(city_osm_dir, "hf.png")))
        seg = np.array(Image.open(os.path.join(city_osm_dir, "seg.png")).convert("P"))
    else:
        raise NotImplementedError

    # Mapping constructions to buildings
    seg[seg == 4] = 2
    # Generate building instance seg maps
    seg, building_stats = get_instance_seg_map(seg)
    return hf.astype(np.int32), seg.astype(np.int32), building_stats


def get_image_patch(image, cx, cy, patch_size):
    sx = cx - patch_size // 2
    sy = cy - patch_size // 2
    ex = sx + patch_size
    ey = sy + patch_size
    return image[sy:ey, sx:ex]


def get_seg_volume(part_hf, part_seg):
    tensor_extruder = extensions.extrude_tensor.TensorExtruder(
        CONSTANTS["LAYOUT_MAX_HEIGHT"]
    )
    seg_volume = tensor_extruder(part_seg, part_hf).squeeze()
    return seg_volume


def get_voxel_intersection_perspective(seg_volume, camera_location):
    CAMERA_FOCAL = (
        CONSTANTS["GES_IMAGE_HEIGHT"] / 2 / np.tan(np.deg2rad(CONSTANTS["GES_VFOV"]))
    )
    # print(seg_volume.size())  # torch.Size([1536, 1536, 640])
    camera_target = {
        "x": seg_volume.size(1) // 2 - 1,
        "y": seg_volume.size(0) // 2 - 1,
    }
    cam_ori_t = torch.tensor(
        [
            camera_location["y"],
            camera_location["x"],
            camera_location["z"],
        ],
        dtype=torch.float32,
        device=seg_volume.device,
    )

    voxel_id, depth2, raydirs = extensions.voxlib.ray_voxel_intersection_perspective(
        seg_volume,
        cam_ori_t,
        torch.tensor(
            [
                camera_target["y"] - camera_location["y"],
                camera_target["x"] - camera_location["x"],
                -camera_location["z"],
            ],
            dtype=torch.float32,
            device=seg_volume.device,
        ),
        torch.tensor([0, 0, 1], dtype=torch.float32),
        CAMERA_FOCAL * 2.06,
        [
            (CONSTANTS["GES_IMAGE_HEIGHT"] - 1) / 2.0,
            (CONSTANTS["GES_IMAGE_WIDTH"] - 1) / 2.0,
        ],
        [CONSTANTS["GES_IMAGE_HEIGHT"], CONSTANTS["GES_IMAGE_WIDTH"]],
        CONSTANTS["N_VOXEL_INTERSECT_SAMPLES"],
    )
    return (
        voxel_id.unsqueeze(dim=0),
        depth2.permute(1, 2, 0, 3, 4).unsqueeze(dim=0),
        raydirs.unsqueeze(dim=0),
        cam_ori_t.unsqueeze(dim=0),
    )


def get_z(device):
    return torch.randn(1, 256, dtype=torch.float32, device=device)


def get_pad_img_bbox(sx, ex, sy, ey):
    psx = sx - CONSTANTS["IMAGE_PADDING"] if sx != 0 else 0
    psy = sy - CONSTANTS["IMAGE_PADDING"] if sy != 0 else 0
    pex = (
        ex + CONSTANTS["IMAGE_PADDING"]
        if ex != CONSTANTS["GES_IMAGE_WIDTH"]
        else CONSTANTS["GES_IMAGE_WIDTH"]
    )
    pey = (
        ey + CONSTANTS["IMAGE_PADDING"]
        if ey != CONSTANTS["GES_IMAGE_HEIGHT"]
        else CONSTANTS["GES_IMAGE_HEIGHT"]
    )
    return psx, pex, psy, pey


def get_img_without_pad(img, sx, ex, sy, ey, psx, pex, psy, pey):
    if CONSTANTS["IMAGE_PADDING"] == 0:
        return img

    return img[
        :,
        :,
        sy - psy : ey - pey if ey != pey else ey,
        sx - psx : ex - pex if ex != pex else ex,
    ]


def render_bg(patch_size, gancraft_bg, hf_seg, voxel_id, depth2, raydirs, cam_ori_t, z):
    _voxel_id = copy.deepcopy(voxel_id)
    _voxel_id[voxel_id >= CONSTANTS["BLD_INS_LABEL_MIN"]] = CONSTANTS[
        "BULIDING_MASK_ID"
    ]
    assert (_voxel_id < CONSTANTS["LAYOUT_N_CLASSES"]).all()
    bg_img = torch.zeros(
        1,
        3,
        CONSTANTS["GES_IMAGE_HEIGHT"],
        CONSTANTS["GES_IMAGE_WIDTH"],
        dtype=torch.float32,
        device=gancraft_bg.output_device,
    )
    # Render background patches by patch to avoid OOM
    for i in range(CONSTANTS["GES_IMAGE_HEIGHT"] // patch_size[0]):
        for j in range(CONSTANTS["GES_IMAGE_WIDTH"] // patch_size[1]):
            sy, sx = i * patch_size[0], j * patch_size[1]
            ey, ex = sy + patch_size[0], sx + patch_size[1]
            psx, pex, psy, pey = get_pad_img_bbox(sx, ex, sy, ey)
            output_bg = gancraft_bg(
                hf_seg,
                _voxel_id[:, psy:pey, psx:pex],
                depth2[:, psy:pey, psx:pex],
                raydirs[:, psy:pey, psx:pex],
                cam_ori_t,
                z,
            )
            bg_img[:, :, sy:ey, sx:ex] = get_img_without_pad(
                output_bg["fake_images"], sx, ex, sy, ey, psx, pex, psy, pey
            )

    return bg_img


def render_fg(
    patch_size,
    gancraft_fg,
    building_id,
    hf_seg,
    voxel_id,
    depth2,
    raydirs,
    cam_ori_t,
    building_stats,
    building_z,
):
    _voxel_id = copy.deepcopy(voxel_id)
    _voxel_id[voxel_id != building_id] = 0
    _voxel_id[voxel_id == building_id] = CONSTANTS["BULIDING_MASK_ID"]
    # assert (_voxel_id < CONSTANTS["LAYOUT_N_CLASSES"]).all()
    _hf_seg = copy.deepcopy(hf_seg)
    _hf_seg[hf_seg != building_id] = 0
    _hf_seg[hf_seg == building_id] = CONSTANTS["BULIDING_MASK_ID"]
    _raydirs = copy.deepcopy(raydirs)
    _raydirs[voxel_id[..., 0, 0] != building_id] = 0

    # Crop the "hf_seg" image using the center of the target building as the reference
    cx = CONSTANTS["LAYOUT_VOL_SIZE"] // 2 - int(building_stats[1])
    cy = CONSTANTS["LAYOUT_VOL_SIZE"] // 2 - int(building_stats[0])
    sx = cx - CONSTANTS["BUILDING_VOL_SIZE"] // 2
    ex = cx + CONSTANTS["BUILDING_VOL_SIZE"] // 2
    sy = cy - CONSTANTS["BUILDING_VOL_SIZE"] // 2
    ey = cy + CONSTANTS["BUILDING_VOL_SIZE"] // 2
    _hf_seg = hf_seg[:, :, sy:ey, sx:ex]

    fg_img = torch.zeros(
        1,
        3,
        CONSTANTS["GES_IMAGE_HEIGHT"],
        CONSTANTS["GES_IMAGE_WIDTH"],
        dtype=torch.float32,
        device=gancraft_fg.output_device,
    )
    fg_mask = torch.zeros(
        1,
        1,
        CONSTANTS["GES_IMAGE_HEIGHT"],
        CONSTANTS["GES_IMAGE_WIDTH"],
        dtype=torch.float32,
        device=gancraft_fg.output_device,
    )
    # Render foreground patches by patch to avoid OOM
    for i in range(CONSTANTS["GES_IMAGE_HEIGHT"] // patch_size[0]):
        for j in range(CONSTANTS["GES_IMAGE_WIDTH"] // patch_size[1]):
            sy, sx = i * patch_size[0], j * patch_size[1]
            ey, ex = sy + patch_size[0], sx + patch_size[1]
            psx, pex, psy, pey = get_pad_img_bbox(sx, ex, sy, ey)

            if torch.count_nonzero(_raydirs[:, sy:ey, sx:ex] > 0):
                output_fg = gancraft_fg(
                    _hf_seg,
                    _voxel_id[:, psy:pey, psx:pex],
                    depth2[:, psy:pey, psx:pex],
                    _raydirs[:, psy:pey, psx:pex],
                    cam_ori_t,
                    torch.from_numpy(np.array(building_stats)).unsqueeze(dim=0),
                    building_z,
                )
                mask = (voxel_id[:, sy:ey, sx:ex, 0, 0] == building_id).unsqueeze(dim=1)
                fg_mask[:, :, sy:ey, sx:ex] = mask
                fg_img[:, :, sy:ey, sx:ex] = mask * get_img_without_pad(
                    output_fg["fake_images"], sx, ex, sy, ey, psx, pex, psy, pey
                )

    return fg_img, fg_mask


def render(
    patch_size,
    seg_volume,
    hf_seg,
    cam_pos,
    gancraft_bg,
    gancraft_fg,
    building_stats,
    bg_z,
    building_zs,
):
    voxel_id, depth2, raydirs, cam_ori_t = get_voxel_intersection_perspective(
        seg_volume, cam_pos
    )
    buildings = torch.unique(voxel_id[voxel_id > CONSTANTS["BLD_INS_LABEL_MIN"]])
    with torch.no_grad():
        bg_img = render_bg(
            patch_size, gancraft_bg, hf_seg, voxel_id, depth2, raydirs, cam_ori_t, bg_z
        )
        for b in buildings:
            fg_img, fg_mask = render_fg(
                patch_size,
                gancraft_fg,
                b.item(),
                hf_seg,
                voxel_id,
                depth2,
                raydirs,
                cam_ori_t,
                building_stats[b.item()],
                building_zs[b.item()],
            )
            bg_img = bg_img * (1 - fg_mask) + fg_img * fg_mask

    return bg_img


def main(
    patch_size,
    output_file,
    gancraft_bg_ckpt,
    gancraft_fg_ckpt,
    sampler_ckpt=None,
    city_osm_dir=None,
):
    # Load checkpoints
    logging.info("Loading checkpoints ...")
    gancraft_bg_ckpt = torch.load(gancraft_bg_ckpt)
    gancraft_fg_ckpt = torch.load(gancraft_fg_ckpt)

    # Initialize models
    logging.info("Initializing models ...")
    gancraft_bg = models.gancraft.GanCraftGenerator(gancraft_bg_ckpt["cfg"])
    gancraft_fg = models.gancraft.GanCraftGenerator(gancraft_fg_ckpt["cfg"])
    if torch.cuda.is_available():
        gancraft_bg = torch.nn.DataParallel(gancraft_bg).cuda()
        gancraft_fg = torch.nn.DataParallel(gancraft_fg).cuda()
    else:
        gancraft_bg.device = torch.device("cpu")
        gancraft_fg.device = torch.device("cpu")

    # Recover from checkpoints
    logging.info("Recovering from checkpoints ...")
    gancraft_bg.load_state_dict(gancraft_bg_ckpt["gancraft_g"])
    gancraft_fg.load_state_dict(gancraft_fg_ckpt["gancraft_g"])

    # Generate height fields and seg maps
    logging.info("Generating city layouts ...")
    hf, seg, building_stats = get_city_layout(city_osm_dir)
    assert hf.shape == seg.shape
    logging.info("City Layout Patch Size (HxW): %s" % (hf.shape,))

    # Generate latent codes
    logging.info("Generating latent codes ...")
    bg_z = get_z(gancraft_bg.output_device)
    building_zs = {
        i + CONSTANTS["BLD_INS_LABEL_MIN"]: get_z(gancraft_bg.output_device)
        for i in range(len(building_stats))
    }

    # TODO: Replace the center crop here
    cy, cx = seg.shape[0] // 2, seg.shape[1] // 2

    # Generate local image patch of the height field and seg map
    part_hf = get_image_patch(hf, cx, cy, CONSTANTS["LAYOUT_VOL_SIZE"])
    part_seg = get_image_patch(seg, cx, cy, CONSTANTS["LAYOUT_VOL_SIZE"])
    assert part_hf.shape == (CONSTANTS["LAYOUT_VOL_SIZE"], CONSTANTS["LAYOUT_VOL_SIZE"])
    assert part_hf.shape == part_seg.shape

    # Recalculate the building positions based on the current patch
    _buildings = np.unique(part_seg[part_seg > CONSTANTS["BLD_INS_LABEL_MIN"]])
    _building_stats = {}
    for b in _buildings:
        _b = b - CONSTANTS["BLD_INS_LABEL_MIN"]
        _building_stats[b] = [
            building_stats[_b, 1] - cy + building_stats[_b, 3] / 2,
            building_stats[_b, 0] - cx + building_stats[_b, 2] / 2,
        ]

    # Build seg_volume
    logging.info("Generating seg volume ...")
    part_hf = torch.from_numpy(part_hf[None, None, ...]).to(gancraft_bg.output_device)
    part_seg = torch.from_numpy(part_seg[None, None, ...]).to(gancraft_bg.output_device)
    # print(part_seg.size())    # torch.Size([1, 1, 1536, 1536])
    assert part_seg.size(1) == 1

    seg_volume = get_seg_volume(part_hf, part_seg)
    part_hf = part_hf / CONSTANTS["LAYOUT_MAX_HEIGHT"]
    part_seg = utils.helpers.masks_to_onehots(
        part_seg[:, 0, :, :], CONSTANTS["LAYOUT_N_CLASSES"]
    )
    hf_seg = torch.cat([part_hf, part_seg], dim=1)
    # print(hf_seg.size())      # torch.Size([1, 8, 1536, 1536])

    # TODO: Generate camera trajectories
    logging.info("Generating camera poses ...")
    cam_pos = [{"x": 767, "y": y, "z": 395} for y in range(517, 117, -20)]

    logging.info("Rendering videos ...")
    video = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        4,
        (CONSTANTS["GES_IMAGE_WIDTH"], CONSTANTS["GES_IMAGE_HEIGHT"]),
    )
    for cp in tqdm(cam_pos):
        try:
            img = render(
                patch_size,
                seg_volume,
                hf_seg,
                cp,
                gancraft_bg,
                gancraft_fg,
                _building_stats,
                bg_z,
                building_zs,
            )
            img = (utils.helpers.tensor_to_image(img, "RGB") * 255).astype(np.uint8)
            video.write(img[..., ::-1])
        except Exception as ex:
            logging.exception(ex)

    video.release()


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (48, 27)
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gancraft_bg_ckpt",
        default=os.path.join(PROJECT_HOME, "output", "gancraft-bg.pth"),
    )
    parser.add_argument(
        "--gancraft_fg_ckpt",
        default=os.path.join(PROJECT_HOME, "output", "gancraft-fg.pth"),
    )
    parser.add_argument(
        "--sampler_ckpt",
        default=os.path.join(PROJECT_HOME, "output", "sampler.pth"),
    )
    parser.add_argument(
        "--city_osm_dir",
        default=os.path.join(PROJECT_HOME, "data", "osm", "US-NewYork"),
    )
    parser.add_argument(
        "--patch_height",
        default=CONSTANTS["GES_IMAGE_HEIGHT"] // 3,
        type=int,
    )
    parser.add_argument(
        "--patch_width",
        default=CONSTANTS["GES_IMAGE_WIDTH"] // 3,
        type=int,
    )
    parser.add_argument(
        "--output_file",
        default=os.path.join(PROJECT_HOME, "output", "rendering.mp4"),
        type=str,
    )
    args = parser.parse_args()

    main(
        (args.patch_height, args.patch_width),
        args.output_file,
        args.gancraft_bg_ckpt,
        args.gancraft_fg_ckpt,
        args.sampler_ckpt,
        args.city_osm_dir,
    )
