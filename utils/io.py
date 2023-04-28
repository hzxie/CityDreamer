# -*- coding: utf-8 -*-
#
# @File:   io.py
# @Author: Haozhe Xie
# @Date:   2019-08-02 10:22:03
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-27 20:23:54
# @Email:  root@haozhexie.com

import io
import numpy as np
import os
import pickle
import sys

from PIL import Image

# Disable the warning message for PIL decompression bomb
# Ref: https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None

from config import cfg

sys.path.append(cfg.MEMCACHED.LIBRARY_PATH)

# References: http://confluence.sensetime.com/pages/viewpage.action?pageId=44650315
mc_client = None
if cfg.MEMCACHED.ENABLED:
    import mc

    mc_client = mc.MemcachedClient.GetInstance(
        cfg.MEMCACHED.SERVER_CONFIG, cfg.MEMCACHED.CLIENT_CONFIG
    )


class IO:
    @classmethod
    def get(cls, file_path):
        if not os.path.exists(file_path):
            return None

        _, file_extension = os.path.splitext(file_path)
        if file_extension in [".png", ".jpg", ".jpeg"]:
            return cls._read_img(file_path)
        if file_extension in [".pkl"]:
            return cls._read_pkl(file_path)
        else:
            raise Exception("Unsupported file extension: %s" % file_extension)

    @classmethod
    def _read_img(cls, file_path):
        if mc_client is None:
            img = Image.open(file_path)
        else:
            pyvector = mc.pyvector()
            mc_client.Get(file_path, pyvector)
            buf = mc.ConvertBuffer(pyvector)
            img = Image.open(io.BytesIO(np.frombuffer(buf, np.uint8)))

        return img

    @classmethod
    def _read_pkl(cls, file_path):
        if mc_client is None:
            with open(file_path, "rb") as f:
                pkl = pickle.load(f)
        else:
            pyvector = mc.pyvector()
            mc_client.Get(file_path, pyvector)
            buf = mc.ConvertBuffer(pyvector)
            pkl = pickle.loads(buf)

        return pkl
