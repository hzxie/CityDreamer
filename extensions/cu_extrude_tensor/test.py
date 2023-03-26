# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2023-03-26 19:23:26
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-03-26 19:36:08
# @Email:  root@haozhexie.com

import logging
import os
import sys
import torch
import unittest

from torch.autograd import gradcheck

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)
from extensions.cu_extrude_tensor import ExtrudeTensorFunction


class ExtrudeTensorTestCase(unittest.TestCase):
    def test_extrude_tensor(self):
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


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    unittest.main()
