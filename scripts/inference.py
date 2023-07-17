# -*- coding: utf-8 -*-
#
# @File:   inference.py
# @Author: Haozhe Xie
# @Date:   2023-05-31 15:01:28
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-07-17 12:43:06
# @Email:  root@haozhexie.com

import argparse
import copy
import cv2
import logging
import math
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
    "EXTENDED_VOL_SIZE": 2880,
    "GES_VFOV": 20,
    "GES_IMAGE_HEIGHT": 540,
    "GES_IMAGE_WIDTH": 960,
    "IMAGE_PADDING": 8,
    "N_SAMPLER_STEPS": 32,
    "N_VOXEL_INTERSECT_SAMPLES": 6,
    "N_TRAJECTORY_POINTS": 24,
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
        vqae.output_device = torch.device("cpu")
        sampler.output_device = torch.device("cpu")
        gancraft_bg.output_device = torch.device("cpu")
        gancraft_fg.output_device = torch.device("cpu")

    # Recover from checkpoints
    logging.info("Recovering from checkpoints ...")
    vqae.load_state_dict(sampler_ckpt["vqae"], strict=False)
    sampler.load_state_dict(sampler_ckpt["sampler"], strict=False)
    gancraft_bg.load_state_dict(gancraft_bg_ckpt["gancraft_g"], strict=False)
    gancraft_fg.load_state_dict(gancraft_fg_ckpt["gancraft_g"], strict=False)

    return vqae, sampler, gancraft_bg, gancraft_fg


def _get_layout_codebook_indexes(sampler, indexes=None):
    with torch.no_grad():
        min_encoding_indices = sampler.module.sample(
            1,
            CONSTANTS["N_SAMPLER_STEPS"],
            x_t=indexes,
            device=sampler.output_device,
        )
        # print(min_encoding_indices.size())  # torch.Size([bs, att_size**2])
        return min_encoding_indices.unsqueeze(dim=2)


def _get_layout(vqae, sampler, min_encoding_indices, codebook):
    with torch.no_grad():
        one_hot = torch.zeros(
            (
                1,
                sampler.module.cfg.NETWORK.SAMPLER.BLOCK_SIZE,
                sampler.module.cfg.NETWORK.VQGAN.N_EMBED,
            ),
            device=sampler.output_device,
        )
        one_hot.scatter_(2, min_encoding_indices, 1)
        # print(min_encoding_indices, torch.argmax(one_hot, dim=2))
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
        return vqae.module.decode(quant)


def generate_city_layout(
    sampler=None,
    vqae=None,
    hf=None,
    seg=None,
    mask=None,
    layout_size=CONSTANTS["EXTENDED_VOL_SIZE"],
):
    OUTPUT_SIZE = vqae.module.cfg.NETWORK.VQGAN.RESOLUTION
    FEATURE_SIZE = vqae.module.cfg.NETWORK.VQGAN.ATTN_RESOLUTION
    SCALE = OUTPUT_SIZE // FEATURE_SIZE
    STRIDE = int(FEATURE_SIZE * 0.75)
    # assert OUTPUT_SIZE == 512
    # assert FEATURE_SIZE == 32
    # assert SCALE == 16

    # Compute the output size and make it compatible to all input sizes as possible
    window_size = int(math.ceil((layout_size / SCALE - FEATURE_SIZE) / STRIDE) + 1)
    code_index_size = (window_size - 1) * STRIDE + FEATURE_SIZE
    output_size = code_index_size * SCALE

    layout = torch.zeros(1, CONSTANTS["LAYOUT_N_CLASSES"], output_size, output_size)
    if hf is None or seg is None:
        lyt_code_idx = (
            torch.ones(
                (1, code_index_size, code_index_size),
                device=sampler.output_device,
            ).long()
            * sampler.module.cfg.NETWORK.VQGAN.N_EMBED
        )
    else:
        raise NotImplementedError

    if mask is not None:
        raise NotImplementedError

    codebook = vqae.module.quantize.get_codebook()
    # Single-pass
    # min_encoding_indices = _get_layout_codebook_indexes(sampler)
    # layout = _get_layout(vqae, sampler, min_encoding_indices, codebook)
    # Multi-pass
    for patch_idx in tqdm(range(window_size**2)):
        i = patch_idx // window_size
        j = patch_idx % window_size
        crs = i * STRIDE
        ccs = j * STRIDE
        _code_idx = lyt_code_idx[
            :, crs : crs + FEATURE_SIZE, ccs : ccs + FEATURE_SIZE
        ].reshape(1, FEATURE_SIZE**2)
        _code_idx = _get_layout_codebook_indexes(sampler, _code_idx)
        _layout = _get_layout(
            vqae,
            sampler,
            _code_idx,
            codebook,
        )
        lrs = crs * SCALE
        lcs = ccs * SCALE
        lyt_code_idx[
            :, crs : crs + FEATURE_SIZE, ccs : ccs + FEATURE_SIZE
        ] = _code_idx.reshape(1, FEATURE_SIZE, FEATURE_SIZE)
        layout[..., lrs : lrs + OUTPUT_SIZE, lcs : lcs + OUTPUT_SIZE] = _layout

    # Crop layout to expected output size
    layout = layout[..., :layout_size, :layout_size]
    assert layout.size(2) == layout.size(3)
    assert layout.size(2) == layout_size

    hf = layout[0, 0] * CONSTANTS["LAYOUT_MAX_HEIGHT"]
    seg = utils.helpers.onehot_to_mask(
        layout[[0], 1:],
        vqae.module.cfg.DATASETS.OSM_LAYOUT.IGNORED_CLASSES,
    ).squeeze()
    return hf.cpu().numpy().astype(np.int32), seg.cpu().numpy().astype(np.int32)


def get_osm_city_layout(city_osm_dir):
    hf = np.array(Image.open(os.path.join(city_osm_dir, "hf.png")))
    seg = np.array(Image.open(os.path.join(city_osm_dir, "seg.png")).convert("P"))
    return hf, seg


def get_city_layout(city_osm_dir=None, sampler=None, vqae=None, hf_seg=None, size=None):
    if city_osm_dir is None:
        hf, seg = generate_city_layout(sampler, vqae, hf_seg, size)
    else:
        hf, seg = get_osm_city_layout(city_osm_dir)

    ins_seg, building_stats = get_instance_seg_map(seg)
    hf[hf >= CONSTANTS["LAYOUT_MAX_HEIGHT"]] = CONSTANTS["LAYOUT_MAX_HEIGHT"] - 1
    return hf.astype(np.int32), ins_seg.astype(np.int32), building_stats


def get_instance_seg_map(seg):
    # Mapping constructions to buildings
    seg[seg == 4] = 2
    # Generate building instance seg maps
    seg, building_stats = scripts.dataset_generator.get_instance_seg_map(seg)
    return seg.astype(np.int32), building_stats


def get_latent_codes(building_stats, bg_style_dim, output_device):
    bg_z = _get_z(output_device, bg_style_dim)
    building_zs = {
        (i + CONSTANTS["BLD_INS_LABEL_MIN"]) * 2: _get_z(output_device)
        for i in range(len(building_stats))
    }
    return bg_z, building_zs


def _get_z(device, z_dim=256):
    if z_dim is None:
        return None
    return torch.randn(1, z_dim, dtype=torch.float32, device=device)


def get_image_patch(image, cx, cy, patch_size):
    sx = cx - patch_size // 2
    sy = cy - patch_size // 2
    ex = sx + patch_size
    ey = sy + patch_size
    return image[sy:ey, sx:ex]


def get_part_hf_seg(hf, seg, cx, cy, patch_size):
    part_hf = get_image_patch(hf, cx, cy, patch_size)
    part_seg = get_image_patch(seg, cx, cy, patch_size)
    assert part_hf.shape == (
        patch_size,
        patch_size,
    ), part_hf.shape
    assert part_hf.shape == part_seg.shape, part_seg.shape
    return part_hf, part_seg


def get_part_building_stats(part_seg, building_stats, cx, cy):
    _buildings = np.unique(part_seg[part_seg > CONSTANTS["BLD_INS_LABEL_MIN"]])
    _building_stats = {}
    for b in _buildings:
        _b = b // 2 - CONSTANTS["BLD_INS_LABEL_MIN"]
        _building_stats[b] = [
            building_stats[_b, 1] - cy + building_stats[_b, 3] / 2,
            building_stats[_b, 0] - cx + building_stats[_b, 2] / 2,
        ]
    return _building_stats


def get_hf_seg_tensor(part_hf, part_seg, output_device):
    part_hf = torch.from_numpy(part_hf[None, None, ...]).to(output_device)
    part_seg = torch.from_numpy(part_seg[None, None, ...]).to(output_device)
    part_hf = part_hf / CONSTANTS["LAYOUT_MAX_HEIGHT"]
    part_seg = utils.helpers.masks_to_onehots(
        part_seg[:, 0, :, :], CONSTANTS["LAYOUT_N_CLASSES"]
    )
    return torch.cat([part_hf, part_seg], dim=1)


def get_seg_volume(part_hf, part_seg):
    if part_hf.shape == (
        CONSTANTS["EXTENDED_VOL_SIZE"],
        CONSTANTS["EXTENDED_VOL_SIZE"],
    ):
        part_hf = part_hf[
            CONSTANTS["BUILDING_VOL_SIZE"] : -CONSTANTS["BUILDING_VOL_SIZE"],
            CONSTANTS["BUILDING_VOL_SIZE"] : -CONSTANTS["BUILDING_VOL_SIZE"],
        ]
        # print(part_hf.shape)  # torch.Size([1, 8, 1536, 1536])
        part_seg = part_seg[
            CONSTANTS["BUILDING_VOL_SIZE"] : -CONSTANTS["BUILDING_VOL_SIZE"],
            CONSTANTS["BUILDING_VOL_SIZE"] : -CONSTANTS["BUILDING_VOL_SIZE"],
        ]
        # print(part_seg.shape)  # torch.Size([1, 8, 1536, 1536])

    assert part_hf.shape == (
        CONSTANTS["LAYOUT_VOL_SIZE"],
        CONSTANTS["LAYOUT_VOL_SIZE"],
    )
    assert part_hf.shape == part_seg.shape, part_seg.shape
    return scripts.dataset_generator.get_seg_volume(part_seg, part_hf)
    # print(seg_volume.size())  # torch.Size([1536, 1536, 640])


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


def get_orbit_camera_positions(radius, altitude):
    camera_positions = []
    cx = CONSTANTS["LAYOUT_VOL_SIZE"] // 2
    cy = cx
    for i in range(CONSTANTS["N_TRAJECTORY_POINTS"]):
        theta = 2 * math.pi / CONSTANTS["N_TRAJECTORY_POINTS"] * i
        cam_x = cx + radius * math.cos(theta)
        cam_y = cy + radius * math.sin(theta)
        camera_positions.append({"x": cam_x, "y": cam_y, "z": altitude})

    return camera_positions


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
    assert hf_seg.size(2) == CONSTANTS["EXTENDED_VOL_SIZE"]
    assert hf_seg.size(3) == CONSTANTS["EXTENDED_VOL_SIZE"]
    hf_seg = hf_seg[
        :,
        :,
        CONSTANTS["BUILDING_VOL_SIZE"] : -CONSTANTS["BUILDING_VOL_SIZE"],
        CONSTANTS["BUILDING_VOL_SIZE"] : -CONSTANTS["BUILDING_VOL_SIZE"],
    ]
    assert hf_seg.size(2) == CONSTANTS["LAYOUT_VOL_SIZE"]
    assert hf_seg.size(3) == CONSTANTS["LAYOUT_VOL_SIZE"]

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
    cx = CONSTANTS["EXTENDED_VOL_SIZE"] // 2 - int(building_stats[1])
    cy = CONSTANTS["EXTENDED_VOL_SIZE"] // 2 - int(building_stats[0])
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
    # Prevent some buildings are out of bound.
    # THIS SHOULD NEVER HAPPEN AGAIN.
    # if (
    #     _hf_seg.size(2) != CONSTANTS["BUILDING_VOL_SIZE"]
    #     or _hf_seg.size(3) != CONSTANTS["BUILDING_VOL_SIZE"]
    # ):
    #     return fg_img, fg_mask

    # Render foreground patches by patch to avoid OOM
    for i in range(CONSTANTS["GES_IMAGE_HEIGHT"] // patch_size[0]):
        for j in range(CONSTANTS["GES_IMAGE_WIDTH"] // patch_size[1]):
            sy, sx = i * patch_size[0], j * patch_size[1]
            ey, ex = sy + patch_size[0], sx + patch_size[1]
            psx, pex, psy, pey = get_pad_img_bbox(sx, ex, sy, ey)

            if torch.count_nonzero(_raydirs[:, sy:ey, sx:ex]) > 0:
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
    hf, seg, building_stats = get_city_layout(None, sampler, vqae)
    # hf, seg, building_stats = get_city_layout(city_osm_dir)
    assert hf.shape == seg.shape
    logging.info("City Layout Patch Size (HxW): %s" % (hf.shape,))

    # Generate latent codes
    logging.info("Generating latent codes ...")
    bg_z, building_zs = get_latent_codes(
        building_stats,
        gancraft_bg.module.cfg.NETWORK.GANCRAFT.STYLE_DIM,
        gancraft_bg.output_device,
    )

    # Simply use image center as the patch center
    cy, cx = seg.shape[0] // 2, seg.shape[1] // 2
    # Generate local image patch of the height field and seg map
    part_hf, part_seg = get_part_hf_seg(hf, seg, cx, cy, CONSTANTS["EXTENDED_VOL_SIZE"])
    # print(part_hf.shape)    # (2880, 2880)
    # print(part_seg.shape)   # (2880, 2880)

    # Recalculate the building positions based on the current patch
    _building_stats = get_part_building_stats(part_seg, building_stats, cx, cy)

    # Generate the concatenated height field and seg. map tensor
    hf_seg = get_hf_seg_tensor(part_hf, part_seg, gancraft_bg.output_device)
    # print(hf_seg.size())    # torch.Size([1, 8, 2880, 2880])

    # Build seg_volume
    logging.info("Generating seg volume ...")
    seg_volume = get_seg_volume(part_hf, part_seg)

    # Generate camera trajectories
    logging.info("Generating camera poses ...")
    radius = np.random.randint(128, 512)
    altitude = np.random.randint(128, 778)
    logging.info("Radius = %d, Altitude = %s" % (radius, altitude))
    cam_pos = get_orbit_camera_positions(radius, altitude)

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
