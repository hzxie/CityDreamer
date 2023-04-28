# -*- coding: utf-8 -*-
#
# @File:   data_transforms.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 14:18:01
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-28 15:20:00
# @Email:  root@haozhexie.com

import numpy as np
import random
import torch

import utils.helpers


class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr["callback"])
            parameters = tr["parameters"] if "parameters" in tr else None
            self.transformers.append(
                {
                    "callback": transformer(parameters, tr["objects"]),
                }
            )

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr["callback"]
            data = transform(data)

        return data


class ToTensor(object):
    def __init__(self, _, objects):
        self.objects = objects

    def __call__(self, data):
        for k, v in data.items():
            if k in self.objects:
                if len(v.shape) == 2:
                    # H, W -> H, W, C
                    v = v[..., None]
                if len(v.shape) == 3:
                    # H, W, C -> C, H, W
                    v = v.transpose((2, 0, 1))

                data[k] = torch.from_numpy(v).float()

        return data


class RandomFlip(object):
    def __init__(self, parameters, objects):
        self.hflip = parameters["hflip"] if parameters else True
        self.vflip = parameters["vflip"] if parameters else True
        self.objects = objects

    def _random_flip(self, img, hflip, vflip):
        if hflip:
            img = np.flip(img, axis=1)
        if vflip:
            img = np.flip(img, axis=0)

        return img.copy()

    def __call__(self, data):
        hflip = True if random.random() <= 0.5 and self.hflip else False
        vflip = True if random.random() <= 0.5 and self.vflip else False
        for k, v in data.items():
            if k in self.objects:
                data[k] = self._random_flip(v, hflip, vflip)

        return data


class CenterCrop(object):
    def __init__(self, parameters, objects):
        self.height = parameters["height"]
        self.width = parameters["width"]
        self.objects = objects

    def _center_crop(self, img):
        h, w = img.shape[0], img.shape[1]
        offset_x = w // 2 - self.width // 2
        offset_y = h // 2 - self.height // 2
        new_img = img[
            offset_y : offset_y + self.height, offset_x : offset_x + self.width
        ]
        return new_img

    def __call__(self, data):
        for k, v in data.items():
            if k in self.objects:
                data[k] = self._center_crop(v)

        return data


class RandomCrop(object):
    def __init__(self, parameters, objects):
        self.height = parameters["height"]
        self.width = parameters["width"]
        self.objects = objects

    def _random_crop(self, img, offset_x, offset_y):
        new_img = None
        new_img = img[
            offset_y : offset_y + self.height, offset_x : offset_x + self.width
        ]
        return new_img

    def __call__(self, data):
        img = data[self.objects[0]]
        h, w = img.shape[0], img.shape[1]
        offset_x = random.randint(0, w - self.width)
        offset_y = random.randint(0, h - self.height)

        for k, v in data.items():
            if k in self.objects:
                data[k] = self._random_crop(v, offset_x, offset_y)

        return data


class ToOneHot(object):
    def __init__(self, parameters, objects):
        self.n_classes = parameters["n_classes"]
        self.ignored_classes = (
            parameters["ignored_classes"] if "ignored_classes" in parameters else []
        )
        self.objects = objects

    def _to_onehot(self, img):
        mask = utils.helpers.mask_to_onehot(img, self.n_classes, self.ignored_classes)
        return mask

    def __call__(self, data):
        for k, v in data.items():
            if k in self.objects:
                data[k] = self._to_onehot(v)

        return data
