# -*- coding: utf-8 -*-
#
# @File:   cam_pose_generator.py
# @Author: Haozhe Xie
# @Date:   2023-06-11 14:44:44
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-06-12 19:39:22
# @Email:  root@haozhexie.com

import argparse
import numpy as np
import pickle
import os

from tqdm import tqdm


def norm_np_arr(arr):
    return arr / np.linalg.norm(arr)


def viewdir_to_rt(eye, at, up):
    zaxis = norm_np_arr(eye - at)
    xaxis = norm_np_arr(np.cross(up, zaxis))
    yaxis = np.cross(zaxis, xaxis)
    c2w = np.array(
        [
            [xaxis[0], yaxis[0], zaxis[0], eye[0]],
            [xaxis[1], yaxis[1], zaxis[1], eye[1]],
            [xaxis[2], yaxis[2], zaxis[2], eye[2]],
            [0, 0, 0, 1],
        ]
    )
    return np.linalg.inv(c2w)


def render_image(voxels, K, Rt, image_size):
    assert voxels.shape == (1536, 1536, 640)
    # Naive Implementation: Project voxels onto image
    image = np.zeros(image_size)
    for x in range(540, 896):
        for y in range(540, 896):
            for z in range(0, 308):
                # Transform voxel coordinates into camera coordinates
                voxel_cam = Rt @ np.array([x, y, z, 1])
                assert voxel_cam[3] == 1
                # voxel_cam = voxel_cam[:3] / voxel_cam[3]
                # Project voxel onto image
                voxel_proj = K @ voxel_cam[:3]
                u, v = voxel_proj[:2] / voxel_proj[2]
                # Interpolate voxel value onto image
                if u >= 0 and u < image_size[1] and v >= 0 and v < image_size[0]:
                    value = voxels[y, x, z]
                    image[int(v), int(u)] = value if value < 10 else 2
    return image


def main(ges_dir, raycasting_dir):
    IMG_SIZE = (540, 960)
    GES_VFOV = 20
    FOCAL_LEN = (IMG_SIZE[0] / 2 / np.tan(np.deg2rad(GES_VFOV))) * 2.06
    CAM_K = np.array(
        [
            [FOCAL_LEN, 0, IMG_SIZE[1] // 2],
            [0, FOCAL_LEN, IMG_SIZE[0] // 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    ge_projects = sorted(os.listdir(ges_dir))
    for gep in tqdm(ge_projects):
        _raycasting_dir = os.path.join(ges_dir, gep, raycasting_dir)
        raycastings = os.listdir(_raycasting_dir)
        for r in raycastings:
            with open(os.path.join(_raycasting_dir, r), "rb") as fp:
                pkl = pickle.load(fp)

            # yxz -> xyz
            eye = pkl["cam_origin"][[1, 0, 2]]
            viewdir = pkl["viewdir"][[1, 0, 2]]
            Rt = viewdir_to_rt(eye, eye + viewdir, [0, 0, 1])
            r = r.replace(gep, "Rt")
            with open(os.path.join(_raycasting_dir, r), "rb") as fp:
                pickle.dump({"K": CAM_K, "Rt": Rt}, fp)

            # Debug: Verify the K and Rt matrices
            # color = render_image(voxels, CAM_K, Rt, (540, 960))
            # plt.imshow(color)
            # plt.show()


if __name__ == "__main__":
    PROJECT_HOME = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir)
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--ges_dir", default=os.path.join(PROJECT_HOME, "data", "ges"))
    parser.add_argument("--rc_dir", default="raycasting")
    args = parser.parse_args()
    main(args.ges_dir, args.rc_dir)
