# -*- coding: utf-8 -*-
#
# @File:   data_loaders.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 10:29:53
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-07 10:39:47
# @Email:  root@haozhexie.com

import numpy as np
import os
import torch

import utils.io
import utils.data_transforms


def get_dataset(cfg, dataset_name, split):
    if dataset_name == "OSM_LAYOUT":
        return OsmLayoutDataset(cfg, split)
    else:
        raise Exception("Unknown dataset: %s" % dataset_name)


def collate_fn(batch):
    data = {}
    for sample in batch:
        for k, v in sample.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return data


class OsmLayoutDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.cities = self._get_cities(cfg, split)
        self.transforms = self._get_data_transforms(cfg, split)

    def __len__(self):
        return len(self.cities)

    def __getitem__(self, idx):
        city = self.cities[idx]
        hf = np.array(utils.io.IO.get(city["hf"])) / 255.0
        seg = np.array(utils.io.IO.get(city["seg"]).convert("P"))
        img = self.transforms(np.stack([hf, seg], axis=2))
        return {"input": img, "output": img}

    def _get_cities(self, cfg, split):
        cities = sorted(os.listdir(cfg.DATASETS.OSM_LAYOUT.DIR))
        cities = cities[:-1] if split == "train" else cities[-1:]
        return [
            {
                "hf": os.path.join(cfg.DATASETS.OSM_LAYOUT.DIR, c, "hf.png"),
                "seg": os.path.join(cfg.DATASETS.OSM_LAYOUT.DIR, c, "seg.png"),
            }
            for c in cities
        ]

    def _get_data_transforms(self, cfg, split):
        if split == "train":
            return utils.data_transforms.Compose(
                [
                    {
                        "callback": "RandomCrop",
                        "parameters": {
                            "height": cfg.NETWORK.VQGAN.RESOLUTION,
                            "width": cfg.NETWORK.VQGAN.RESOLUTION,
                        },
                    },
                    {"callback": "RandomFlip", "parameters": None},
                    {
                        "callback": "ToOneHot",
                        "parameters": {
                            "n_classes": cfg.DATASETS.OSM_LAYOUT.N_CLASSES,
                            "ignored_classes": cfg.DATASETS.OSM_LAYOUT.IGNORED_CLASSES,
                        },
                    },
                    {"callback": "ToTensor", "parameters": None},
                ]
            )
        else:
            return utils.data_transforms.Compose(
                [
                    {
                        "callback": "CenterCrop",
                        "parameters": {
                            "height": cfg.NETWORK.VQGAN.RESOLUTION,
                            "width": cfg.NETWORK.VQGAN.RESOLUTION,
                        },
                    },
                    {
                        "callback": "ToOneHot",
                        "parameters": {
                            "n_classes": cfg.DATASETS.OSM_LAYOUT.N_CLASSES,
                            "ignored_classes": cfg.DATASETS.OSM_LAYOUT.IGNORED_CLASSES,
                        },
                    },
                    {"callback": "ToTensor", "parameters": None},
                ]
            )
