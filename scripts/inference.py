# -*- coding: utf-8 -*-
#
# @File:   inference.py
# @Author: Haozhe Xie
# @Date:   2023-05-31 15:01:28
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-07-14 20:05:16
# @Email:  root@haozhexie.com

import argparse
import copy
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.transforms
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
import scripts.dataset_generator
import utils.helpers

CONSTANTS = {
    "ROAD_ID": 1,
    "BLD_FACADE_ID": 2,
    "BLD_ROOF_ID": 7,
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


def get_models(sampler_ckpt, gancraft_bg_ckpt, gancraft_fg_ckpt):
    # Load checkpoints
    logging.info("Loading checkpoints ...")
    sampler_ckpt = torch.load(sampler_ckpt)
    gancraft_bg_ckpt = torch.load(gancraft_bg_ckpt)
    gancraft_fg_ckpt = torch.load(gancraft_fg_ckpt)

    # Initialize models
    vqae = models.vqgan.VQAutoEncoder(sampler_ckpt["cfg"])
    sampler = models.sampler.AbsorbingDiffusionSampler(sampler_ckpt["cfg"])
    gancraft_bg = models.gancraft.GanCraftGenerator(gancraft_bg_ckpt["cfg"])
    gancraft_fg = models.gancraft.GanCraftGenerator(gancraft_fg_ckpt["cfg"])
    if torch.cuda.is_available():
        vqae = torch.nn.DataParallel(vqae).cuda()
        sampler = torch.nn.DataParallel(sampler).cuda()
        gancraft_bg = torch.nn.DataParallel(gancraft_bg).cuda()
        gancraft_fg = torch.nn.DataParallel(gancraft_fg).cuda()
    else:
        vqae.device = torch.device("cpu")
        sampler.device = torch.device("cpu")
        gancraft_bg.device = torch.device("cpu")
        gancraft_fg.device = torch.device("cpu")

    # Recover from checkpoints
    logging.info("Recovering from checkpoints ...")
    vqae.load_state_dict(sampler_ckpt["vqae"], strict=False)
    sampler.load_state_dict(sampler_ckpt["sampler"], strict=False)
    gancraft_bg.load_state_dict(gancraft_bg_ckpt["gancraft_g"], strict=False)
    gancraft_fg.load_state_dict(gancraft_fg_ckpt["gancraft_g"], strict=False)

    return vqae, sampler, gancraft_bg, gancraft_fg


def get_layout_codebook_indexes(sampler, vqae, indexes=None, temperature=1):
    with torch.no_grad():
        min_encoding_indices = sampler.module.sample(
            1,
            sampler.module.cfg.NETWORK.SAMPLER.TOTAL_STEPS,
            x_t=indexes,
            temperature=temperature,
            device=sampler.device,
        )
        # print(min_encoding_indices.size())  # torch.Size([bs, att_size**2])
        min_encoding_indices = min_encoding_indices.unsqueeze(dim=2)
        one_hot = torch.zeros(
            (
                1,
                sampler.module.cfg.NETWORK.SAMPLER.BLOCK_SIZE,
                vqae.module.cfg.NETWORK.VQGAN.N_EMBED,
            ),
            device=sampler.device,
        )
        return one_hot.scatter_(2, min_encoding_indices, 1)


def get_layout(vqae, one_hot):
    with torch.no_grad():
        codebook = vqae.module.quantize.get_codebook()
        quant = (
            torch.matmul(
                one_hot.view(-1, vqae.module.cfg.NETWORK.VQGAN.N_EMBED), codebook
            )
            .float()
            .view(
                1,
                vqae.module.cfg.NETWORK.VQGAN.ATTN_RESOLUTION,
                vqae.module.cfg.NETWORK.VQGAN.ATTN_RESOLUTION,
                vqae.module.cfg.NETWORK.VQGAN.EMBED_DIM,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        # print(quant.size())   # torch.Size([bs, embed_dim, att_size, att_size])
        pred = vqae.module.decode(quant)


def get_city_layout(city_osm_dir=None, sampler=None, vqae=None, hf_seg=None):
    if city_osm_dir is not None:
        hf = np.array(Image.open(os.path.join(city_osm_dir, "hf.png")))
        seg = np.array(Image.open(os.path.join(city_osm_dir, "seg.png")).convert("P"))
    else:
        size = (CONSTANTS["LAYOUT_VOL_SIZE"],) * 2 if hf_seg is None else hf_seg.shape

    # Mapping constructions to buildings
    seg[seg == 4] = 2
    # Generate building instance seg maps
    seg, building_stats = scripts.dataset_generator.get_instance_seg_map(seg)
    return hf.astype(np.int32), seg.astype(np.int32), building_stats


def get_image_patch(image, cx, cy, patch_size):
    sx = cx - patch_size // 2
    sy = cy - patch_size // 2
    ex = sx + patch_size
    ey = sy + patch_size
    return image[sy:ey, sx:ex]


def get_part_hf_seg(hf, seg, cx, cy):
    part_hf = get_image_patch(hf, cx, cy, CONSTANTS["LAYOUT_VOL_SIZE"])
    part_seg = get_image_patch(seg, cx, cy, CONSTANTS["LAYOUT_VOL_SIZE"])
    assert part_hf.shape == (CONSTANTS["LAYOUT_VOL_SIZE"], CONSTANTS["LAYOUT_VOL_SIZE"]), part_hf.shape
    assert part_hf.shape == part_seg.shape, part_seg.shape
    return part_hf, part_seg


def get_voxel_intersection_perspective(seg_volume, camera_location):
    CAMERA_FOCAL = (
        CONSTANTS["GES_IMAGE_HEIGHT"] / 2 / np.tan(np.deg2rad(CONSTANTS["GES_VFOV"]))
    )
    # print(seg_volume.size())  # torch.Size([1536, 1536, 640])
    camera_target = {
        "x": seg_volume.size(1) // 2 - 1,
        "y": seg_volume.size(0) // 2 - 1,
    }
    cam_origin = torch.tensor(
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
        cam_origin,
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
        cam_origin.unsqueeze(dim=0),
    )


def get_z(device, z_dim=256):
    if z_dim is None:
        return None
    return torch.randn(1, z_dim, dtype=torch.float32, device=device)


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


def render_bg(
    patch_size, gancraft_bg, hf_seg, voxel_id, depth2, raydirs, cam_origin, z
):
    blurrer = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(2, 2))
    _voxel_id = copy.deepcopy(voxel_id)
    _voxel_id[voxel_id >= CONSTANTS["BLD_INS_LABEL_MIN"]] = CONSTANTS["BLD_FACADE_ID"]
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
                hf_seg=hf_seg,
                voxel_id=_voxel_id[:, psy:pey, psx:pex],
                depth2=depth2[:, psy:pey, psx:pex],
                raydirs=raydirs[:, psy:pey, psx:pex],
                cam_origin=cam_origin,
                building_stats=None,
                z=z,
                deterministic=True,
            )
            # Make road blurry
            road_mask = (
                (_voxel_id[:, None, psy:pey, psx:pex, 0, 0] == CONSTANTS["ROAD_ID"])
                .repeat(1, 3, 1, 1)
                .float()
            )
            output_bg = blurrer(output_bg) * road_mask + output_bg * (1 - road_mask)
            bg_img[:, :, sy:ey, sx:ex] = get_img_without_pad(
                output_bg, sx, ex, sy, ey, psx, pex, psy, pey
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
    cam_origin,
    building_stats,
    building_z,
):
    _voxel_id = copy.deepcopy(voxel_id)
    _curr_bld = torch.tensor([building_id, building_id - 1], device=voxel_id.device)
    _voxel_id[~torch.isin(_voxel_id, _curr_bld)] = 0
    _voxel_id[voxel_id == building_id] = CONSTANTS["BLD_FACADE_ID"]
    _voxel_id[voxel_id == building_id - 1] = CONSTANTS["BLD_ROOF_ID"]

    # assert (_voxel_id < CONSTANTS["LAYOUT_N_CLASSES"]).all()
    _hf_seg = copy.deepcopy(hf_seg)
    _hf_seg[hf_seg != building_id] = 0
    _hf_seg[hf_seg == building_id] = CONSTANTS["BLD_FACADE_ID"]
    _raydirs = copy.deepcopy(raydirs)
    _raydirs[_voxel_id[..., 0, 0] == 0] = 0

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
    # Prevent some buildings are out of bound
    if (
        _hf_seg.size(2) != CONSTANTS["BUILDING_VOL_SIZE"]
        or _hf_seg.size(3) != CONSTANTS["BUILDING_VOL_SIZE"]
    ):
        return fg_img, fg_mask

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
                    cam_origin,
                    building_stats=torch.from_numpy(np.array(building_stats)).unsqueeze(
                        dim=0
                    ),
                    z=building_z,
                    deterministic=True,
                )
                facade_mask = (
                    voxel_id[:, sy:ey, sx:ex, 0, 0] == building_id
                ).unsqueeze(dim=1)
                roof_mask = (
                    voxel_id[:, sy:ey, sx:ex, 0, 0] == building_id - 1
                ).unsqueeze(dim=1)
                facade_img = facade_mask * get_img_without_pad(
                    output_fg, sx, ex, sy, ey, psx, pex, psy, pey
                )
                # Make roof blurry
                # output_fg = F.interpolate(
                #     F.interpolate(output_fg * 0.8, scale_factor=0.75),
                #     scale_factor=4 / 3,
                # ),
                roof_img = roof_mask * get_img_without_pad(
                    output_fg,
                    sx,
                    ex,
                    sy,
                    ey,
                    psx,
                    pex,
                    psy,
                    pey,
                )
                fg_mask[:, :, sy:ey, sx:ex] = torch.logical_or(facade_mask, roof_mask)
                fg_img[:, :, sy:ey, sx:ex] = (
                    facade_img * facade_mask + roof_img * roof_mask
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
    voxel_id, depth2, raydirs, cam_origin = get_voxel_intersection_perspective(
        seg_volume, cam_pos
    )
    buildings = torch.unique(voxel_id[voxel_id > CONSTANTS["BLD_INS_LABEL_MIN"]])
    # Remove odd numbers from the list because they are reserved by roofs.
    buildings = buildings[buildings % 2 == 0]
    with torch.no_grad():
        bg_img = render_bg(
            patch_size, gancraft_bg, hf_seg, voxel_id, depth2, raydirs, cam_origin, bg_z
        )
        for b in buildings:
            assert b % 2 == 0, "Building Instance ID MUST be an even number."
            fg_img, fg_mask = render_fg(
                patch_size,
                gancraft_fg,
                b.item(),
                hf_seg,
                voxel_id,
                depth2,
                raydirs,
                cam_origin,
                building_stats[b.item()],
                building_zs[b.item()],
            )
            bg_img = bg_img * (1 - fg_mask) + fg_img * fg_mask

    return bg_img


def get_video(frames, output_file):
    video = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"avc1"),
        4,
        (CONSTANTS["GES_IMAGE_WIDTH"], CONSTANTS["GES_IMAGE_HEIGHT"]),
    )
    for f in frames:
        video.write(f)

    video.release()


def main(
    patch_size,
    output_file,
    gancraft_bg_ckpt,
    gancraft_fg_ckpt,
    sampler_ckpt=None,
    city_osm_dir=None,
):
    vqae, sampler, gancraft_bg, gancraft_fg = get_models(
        sampler_ckpt, gancraft_bg_ckpt, gancraft_fg_ckpt
    )
    # Generate height fields and seg maps
    logging.info("Generating city layouts ...")
    # hf, seg, building_stats = get_city_layout(None, sampler, vqae)
    hf, seg, building_stats = get_city_layout(city_osm_dir)
    assert hf.shape == seg.shape
    logging.info("City Layout Patch Size (HxW): %s" % (hf.shape,))

    # Generate latent codes
    logging.info("Generating latent codes ...")
    bg_z = get_z(
        gancraft_bg.output_device, gancraft_bg.module.cfg.NETWORK.GANCRAFT.STYLE_DIM
    )
    building_zs = {
        (i + CONSTANTS["BLD_INS_LABEL_MIN"]) * 2: get_z(gancraft_bg.output_device)
        for i in range(len(building_stats))
    }

    # Simply use image center as the patch center
    cy, cx = seg.shape[0] // 2, seg.shape[1] // 2
    # Generate local image patch of the height field and seg map
    part_hf, part_seg = get_part_hf_seg(hf, seg, cx, cy)

    # Build seg_volume
    logging.info("Generating seg volume ...")
    seg_volume = scripts.dataset_generator.get_seg_volume(part_seg, part_hf)

    # Recalculate the building positions based on the current patch
    _buildings = np.unique(part_seg[part_seg > CONSTANTS["BLD_INS_LABEL_MIN"]])
    _building_stats = {}
    for b in _buildings:
        _b = b // 2 - CONSTANTS["BLD_INS_LABEL_MIN"]
        _building_stats[b] = [
            building_stats[_b, 1] - cy + building_stats[_b, 3] / 2,
            building_stats[_b, 0] - cx + building_stats[_b, 2] / 2,
        ]

    part_hf = torch.from_numpy(part_hf[None, None, ...]).to(gancraft_bg.output_device)
    part_seg = torch.from_numpy(part_seg[None, None, ...]).to(gancraft_bg.output_device)
    part_hf = part_hf / CONSTANTS["LAYOUT_MAX_HEIGHT"]
    part_seg = utils.helpers.masks_to_onehots(
        part_seg[:, 0, :, :], CONSTANTS["LAYOUT_N_CLASSES"]
    )
    hf_seg = torch.cat([part_hf, part_seg], dim=1)
    # print(hf_seg.size())      # torch.Size([1, 8, 1536, 1536])

    # TODO: Generate camera trajectories
    logging.info("Generating camera poses ...")
    cam_pos = [{"x": 767, "y": y, "z": 354} for y in range(517, 117, -20)]

    logging.info("Rendering videos ...")
    frames = []
    for cp in tqdm(cam_pos):
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
        frames.append(img[..., ::-1])
        # cv2.imwrite("output/test.jpg", img[..., ::-1])

    get_video(frames, output_file)


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
        default=CONSTANTS["GES_IMAGE_HEIGHT"] // 5,
        type=int,
    )
    parser.add_argument(
        "--patch_width",
        default=CONSTANTS["GES_IMAGE_WIDTH"] // 5,
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
