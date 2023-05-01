# -*- coding: utf-8 -*-
#
# @File:   footage_logo_removal.py
# @Author: Haozhe Xie
# @Date:   2023-05-01 11:18:46
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-05-01 18:30:17
# @Email:  root@haozhexie.com
#
# Quick Start
# - pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# - pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
# - git clone https://github.com/MCG-NKU/E2FGVI.git
# - [MANUAL DOWNLOAD CKPT to E2FGVI/release_model]
# - https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3/view
#
# References
# - https://colab.research.google.com/drive/12rwY2gtG8jVWlNx9pjmmM8uGmh5ue18G

import argparse
import importlib
import numpy as np
import os
import sys
import torch

from PIL import Image
from tqdm import tqdm

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)


def get_e2fgvi_model(e2fgvi_home, ckpt_file_path):
    sys.path.append(e2fgvi_home)
    model = importlib.import_module("model.e2fgvi_hq")
    model = model.InpaintGenerator()
    if torch.cuda.is_available():
        model = model.cuda()

    ckpt = torch.load(ckpt_file_path)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def get_frames(files):
    frames = []
    for f in files:
        frame = Image.open(f)
        frames.append(np.array(frame) / 255.0 * 2 - 1)

    return np.stack(frames, axis=0).transpose((0, 3, 1, 2))


def _get_ref_indexes(frame_idx, neighbor_ids, n_ref, ref_step, n_frames):
    ref_index = []
    if n_ref == -1:
        for i in range(0, n_frames, ref_step):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, frame_idx - ref_step * (n_ref // 2))
        end_idx = min(n_frames, frame_idx + ref_step * (n_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_step):
            if i not in neighbor_ids:
                if len(ref_index) > n_ref:
                    break
                ref_index.append(i)

    return ref_index


def get_inpainted_frames(frames, masks, model):
    N_REF = -1
    REF_STEP = 10
    NEIGHBOR_STRIDE = 5
    MOD_SIZE_H = 60
    MOD_SIZE_W = 108

    n_frames = len(frames)
    ip_frames = [None] * n_frames
    for f_idx in tqdm(range(0, n_frames, NEIGHBOR_STRIDE)):
        neighbor_ids = [
            i
            for i in range(
                max(0, f_idx - NEIGHBOR_STRIDE),
                min(n_frames, f_idx + NEIGHBOR_STRIDE + 1),
            )
        ]
        ref_ids = _get_ref_indexes(f_idx, neighbor_ids, N_REF, REF_STEP, n_frames)
        selected_imgs = frames[neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks[neighbor_ids + ref_ids, :, :, :]
        h, w = frames.size(2), frames.size(3)

        with torch.no_grad():
            masked_imgs = selected_imgs * (1 - selected_masks)
            masked_imgs = masked_imgs.unsqueeze(dim=0)
            h_pad = (MOD_SIZE_H - h % MOD_SIZE_H) % MOD_SIZE_H
            w_pad = (MOD_SIZE_W - w % MOD_SIZE_W) % MOD_SIZE_W
            masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [3])], 3)[
                :, :, :, : h + h_pad, :
            ]
            masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [4])], 4)[
                :, :, :, :, : w + w_pad
            ]
            if torch.cuda.is_available():
                masked_imgs = masked_imgs.cuda()

            pred_img, _ = model(masked_imgs, len(neighbor_ids))
            if h_pad != 0:
                half_h_pad = h_pad // 2
                pred_img = pred_img[:, :, half_h_pad:-half_h_pad, :]
            if w_pad != 0:
                half_w_pad = w_pad // 2
                pred_img = pred_img[:, :, :, half_w_pad:-half_w_pad]

            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = pred_img[i].cpu() * masks[idx] + frames[idx] * (1 - masks[idx])
                img = (img + 1) / 2 * 255
                if ip_frames[idx] is None:
                    ip_frames[idx] = img
                else:
                    ip_frames[idx] = ip_frames[idx].float() * 0.5 + img.float() * 0.5

    return [ipf.permute(1, 2, 0).numpy() for ipf in ip_frames]


def main(e2fgvi_home, ckpt_file_path, ges_dir):
    # Set up E2FGVI model
    model = get_e2fgvi_model(e2fgvi_home, ckpt_file_path)
    # Generate the mask for Google Earth logo for 960x540 images
    ge_logo_mask = np.zeros((1, 1, 540, 960), bool)
    ge_logo_mask[:, :, 488:508, 842:940] = 1
    # Inpainting images for the footage images
    ge_projects = sorted(os.listdir(ges_dir))
    for gep in tqdm(ge_projects):
        files = sorted(os.listdir(os.path.join(ges_dir, gep, "footage")))
        frames = get_frames([os.path.join(ges_dir, gep, "footage", f) for f in files])
        masks = ge_logo_mask.repeat(len(frames), axis=0)
        frames = torch.from_numpy(frames).float()
        masks = torch.from_numpy(masks).float()
        new_frames = get_inpainted_frames(frames, masks, model)
        for fn, nf in zip(files, new_frames):
            Image.fromarray(nf.astype(np.uint8)).save(
                os.path.join(ges_dir, gep, "footage", fn)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--e2fgvi_home",
        help="The path to the project dir of E2FGVI",
        default=os.path.join(PROJECT_HOME, os.pardir, "E2FGVI"),
    )
    parser.add_argument(
        "--e2fgvi_ckpt",
        help="The path to the checkpoint of SAM",
        default=os.path.join(
            PROJECT_HOME,
            os.pardir,
            "E2FGVI",
            "release_model",
            "E2FGVI-HQ-CVPR22.pth",
        ),
    )
    parser.add_argument("--ges_dir", default=os.path.join(PROJECT_HOME, "data", "ges"))
    args = parser.parse_args()

    main(args.e2fgvi_home, args.e2fgvi_ckpt, args.ges_dir)
