# -*- coding: utf-8 -*-
#
# @File:   data_transforms.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 14:18:01
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-07 10:49:52
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
        if type(img) == list:
            return torch.from_numpy(np.array(img)).float().permute(0, 3, 1, 2)
        else:
            return torch.from_numpy(img).permute(2, 0, 1).float()


class RandomFlip(object):
    def __init__(self, parameters):
        self.hflip = parameters["hflip"] if parameters else True
        self.vflip = parameters["vflip"] if parameters else True

    def _random_flip(self, img, hflip, vflip):
        if hflip:
            img = np.flip(img, axis=1)
        if vflip:
            img = np.flip(img, axis=0)

        return img.copy()

    def __call__(self, img):
        hflip = True if random.random() <= 0.5 and self.hflip else False
        vflip = True if random.random() <= 0.5 and self.vflip else False
        if type(img) == list:
            return [self._random_flip(i, hflip, vflip) for i in img]
        else:
            return self._random_flip(img, hflip, vflip)


class CenterCrop(object):
    def __init__(self, parameters):
        self.height = parameters["height"]
        self.width = parameters["width"]

    def _center_crop(self, img):
        h, w, _ = img.shape
        offset_x = w // 2 - self.width // 2
        offset_y = h // 2 - self.height // 2
        new_img = img[
            offset_y : offset_y + self.height, offset_x : offset_x + self.width, :
        ]
        return new_img

    def __call__(self, img):
        if type(img) == list:
            return [self._center_crop(i) for i in img]
        else:
            return self._center_crop(img)


class RandomCrop(object):
    def __init__(self, parameters):
        self.height = parameters["height"]
        self.width = parameters["width"]

    def _random_crop(self, img):
        h, w, _ = img.shape
        new_img = None
        offset_x = random.randint(0, w - self.width)
        offset_y = random.randint(0, h - self.height)
        new_img = img[
            offset_y : offset_y + self.height, offset_x : offset_x + self.width, :
        ]
        return new_img

    def __call__(self, img):
        if type(img) == list:
            return [self._random_crop(i) for i in img]
        else:
            return self._random_crop(img)


class ToOneHot(object):
    def __init__(self, parameters):
        self.n_classes = parameters["n_classes"]
        self.ignored_classes = parameters["ignored_classes"]

    def _to_onehot(self, img):
        assert img.shape[2] == 2
        mask = utils.helpers.mask_to_onehot(
            img[..., 1], self.n_classes, self.ignored_classes
        )
        return np.concatenate([img[..., 0][..., None], mask], axis=2)

    def __call__(self, img):
        if type(img) == list:
            return [self._to_onehot(i) for i in img]
        else:
            return self._to_onehot(img)
