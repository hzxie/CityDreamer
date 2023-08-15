# -*- coding: utf-8 -*-
#
# @File:   camera_error.py
# @Author: Zhaoxi Chen (@FrozenBurning)
# @Date:   2023-08-13 15:02:56
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-08-15 19:37:25
# @Email:  root@haozhexie.com

import argparse
import collections
import logging
import numpy as np
import scipy
import torch
import torch.nn.functional as F


BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return self._qvec2rotmat(self.qvec)

    def _qvec2rotmat(self, qvec):
        return np.array(
            [
                [
                    1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                    2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                    2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
                ],
                [
                    2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                    1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                    2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
                ],
                [
                    2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                    2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                    1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
                ],
            ]
        )


def get_colmap_cam_pose(txt_file_path):
    """
    See also: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(txt_file_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def affine_registration(P, Q):
    """
    Inputs:
        - P, a (n,dim) [or (dim,n)] matrix, a point cloud of n points in dim dimension.
        - Q, a (n,dim) [or (dim,n)] matrix, a point cloud of n points in dim dimension.
        P and Q must be of the same shape.
    Returns:
        - Pt, the P point cloud, transformed to fit to Q
        - (T,t) the affine transform
    """
    transposed = False
    if P.shape[0] < P.shape[1]:
        transposed = True
        P = P.T
        Q = Q.T

    (n, dim) = P.shape
    # Compute least squares
    p, res, rnk, s = scipy.linalg.lstsq(np.hstack((P, np.ones([n, 1]))), Q)
    scaled_res = res**0.5 / Q.max(0)
    # Get translation
    t = p[-1].T
    # Get transform matrix
    T = p[:-1].T
    # Compute transformed pointcloud
    Pt = P @ T.T + t
    if transposed:
        Pt = Pt.T

    return Pt, (T, t)


def main(est_cam_pose, gt_cam_pose):
    logging.info("Reading files ...")
    gt_bin = np.load(gt_cam_pose, allow_pickle=True)
    colmap_txt = get_colmap_cam_pose(est_cam_pose)

    gt_poses = []
    est_cam_poses = []
    for id in colmap_txt.keys():
        _gt = gt_bin[id - 1]
        gt_poses.append([_gt["z"], _gt["x"], _gt["y"]])

        R = colmap_txt[id].qvec2rotmat()
        T = colmap_txt[id].tvec
        _pose = -R.T @ T
        est_cam_poses.append(_pose)

    logging.info("Normalizing poses ...")
    gt_poses = np.stack(gt_poses)
    est_cam_poses = np.stack(est_cam_poses)
    gt_poses = gt_poses / gt_poses.max()
    est_cam_poses = est_cam_poses / est_cam_poses.max()

    n_cam_poses = gt_poses.shape[0]
    logging.debug("The number of camera poses: %d" % n_cam_poses)

    logging.info("Registration ...")
    tr_gt_poses, _ = affine_registration(gt_poses, est_cam_poses)
    cam_error = (
        F.mse_loss(
            torch.from_numpy(tr_gt_poses.T), torch.from_numpy(est_cam_poses.T)
        ).item()
        * n_cam_poses
    )
    logging.info("Camera Error: %.6f" % cam_error)

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(gt_poses[:, 0], gt_poses[:, 1], gt_poses[:, 2], color="orange")
    # ax.scatter3D(tr_gt_poses[:, 0], tr_gt_poses[:, 1], tr_gt_poses[:, 2], color="green")
    # ax.scatter3D(
    #     est_cam_poses[:, 0], est_cam_poses[:, 1], est_cam_poses[:, 2], color="blue"
    # )
    # ax.scatter3D(
    #     est_cam_poses[0, 0], est_cam_poses[0, 1], est_cam_poses[0, 2], color="red"
    # )
    # ax.scatter3D(
    #     tr_gt_poses[0, 0], tr_gt_poses[0, 1], tr_gt_poses[0, 2], color="yellow"
    # )
    # plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--est",
        "-e",
        help="The images.txt file generated by colmap",
        type=str,
        required=True,
    )
    parser.add_argument("--gt", "-g", type=str, required=True)
    args = parser.parse_args()

    main(args.est, args.gt)
