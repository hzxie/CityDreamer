# -*- coding: utf-8 -*-
#
# @File:   data_transforms.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 14:18:01
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-06 15:43:40
# @Email:  root@haozhexie.com

import numpy as np
import random
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr["callback"])
            parameters = tr["parameters"] if "parameters" in tr else None
            self.transformers.append(
                {
                    "callback": transformer(parameters),
                }
            )

    def __call__(self, img):
        for tr in self.transformers:
            transform = tr["callback"]
            img = transform(img)

        return img


class ToTensor(object):
    def __init__(self, _):
        pass

    def __call__(self, img):
        return torch.from_numpy(img).permute(2, 0, 1).float()


class RandomFlip(object):
    def __init__(self, _):
        pass

    def __call__(self, img):
        if random.random() <= 0.5:
            img = np.flip(img, axis=1)

        if random.random() <= 0.5:
            img = np.flip(img, axis=0)

        return img.copy()


class CenterCrop(object):
    def __init__(self, parameters):
        self.height = parameters["height"]
        self.width = parameters["width"]

    def __call__(self, img):
        h, w, _ = img.shape
        offset_x = w // 2 - self.width // 2
        offset_y = h // 2 - self.height // 2
        new_img = img[
            offset_y : offset_y + self.height, offset_x : offset_x + self.width, :
        ]

        return new_img


class RandomCrop(object):
    def __init__(self, parameters):
        self.height = parameters["height"]
        self.width = parameters["width"]
        self.entropy_limit = (
            parameters["entropy_limit"] if "entropy_limit" in parameters else -1
        )

    def __call__(self, img):
        h, w, _ = img.shape
        new_img = None
        while True:
            offset_x = random.randint(0, w - self.width)
            offset_y = random.randint(0, h - self.height)
            new_img = img[
                offset_y : offset_y + self.height, offset_x : offset_x + self.width, :
            ]
            # TODO: Entropy check
            break

        return new_img
