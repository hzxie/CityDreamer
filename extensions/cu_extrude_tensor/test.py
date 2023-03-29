# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2023-03-26 19:23:26
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-03-29 16:09:23
# @Email:  root@haozhexie.com

import logging
import mayavi.mlab
import numpy as np
import os
import sys
import torch
import unittest

from PIL import Image
from torch.autograd import gradcheck

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)
from extensions.cu_extrude_tensor import ExtrudeTensorFunction


class ExtrudeTensorTestCase(unittest.TestCase):
    def test_extrude_tensor_grad(self):
        # Make sure that the int types are replaced by double in CUDA implementation
        SIZE = 16
        seg_map = (
            torch.randint(low=1, high=7, size=(SIZE, SIZE))
            .double()
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
        )
        height_field = (
            torch.randint(low=0, high=255, size=(SIZE, SIZE))
            .double()
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
        )
        logging.debug("SegMap Size: %s" % (seg_map.size(),))
        logging.debug("HeightField Size: %s" % (height_field.size(),))
        seg_map.requires_grad = True
        height_field.requires_grad = True
        logging.info(
            "Gradient Check: %s" % "OK"
            if gradcheck(
                ExtrudeTensorFunction.apply, [seg_map.cuda(), height_field.cuda(), 256]
            )
            else "Failed"
        )

    def test_extrude_tensor_gen(self):
        MAX_HEIGHT = 256
        proj_home_dir = os.path.join(
            os.path.dirname(__file__), os.path.pardir, os.path.pardir
        )
        osm_data_dir = os.path.join(proj_home_dir, "data", "osm")
        osm_name = "NewYork-4km"
        seg_map = Image.open(
            os.path.join(osm_data_dir, "%s-seg.png" % osm_name)
        ).convert("P")
        height_field = Image.open(os.path.join(osm_data_dir, "%s-hf.png" % osm_name))

        seg_map_tnsr = (
            torch.from_numpy(np.array(seg_map))
            .unsqueeze(dim=0)
            .int()
            .unsqueeze(dim=0)
            .cuda()
        )
        height_field_tnsr = (
            torch.from_numpy(np.array(height_field))
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
            .int()
            .cuda()
        )
        volume = ExtrudeTensorFunction.apply(
            seg_map_tnsr, height_field_tnsr, MAX_HEIGHT
        )
        # 3D Visualization
        vol = volume.squeeze().cpu().numpy().astype(np.uint8)
        vol = vol[1000:1250, 1000:1250]

        x, y, z = np.where(vol != 0)
        n_pts = len(x)
        colors = np.zeros((n_pts, 4), dtype=np.uint8)
        colors[vol[x, y, z] == 1] = [128, 0, 0, 255]
        colors[vol[x, y, z] == 2] = [0, 128, 0, 255]
        colors[vol[x, y, z] == 3] = [128, 128, 0, 255]
        colors[vol[x, y, z] == 4] = [0, 0, 128, 255]
        colors[vol[x, y, z] == 5] = [128, 0, 128, 255]
        colors[vol[x, y, z] == 6] = [0, 128, 128, 255]

        mayavi.mlab.figure(size=(1600, 900), bgcolor=(1, 1, 1))
        pts = mayavi.mlab.points3d(x, y, z, mode="cube", scale_factor=1)
        pts.glyph.scale_mode = "scale_by_vector"
        pts.mlab_source.dataset.point_data.scalars = colors
        mayavi.mlab.show()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    unittest.main()