# -*- coding: utf-8 -*-
#
# @File:   data_loaders.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 10:29:53
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-27 17:26:27
# @Email:  root@haozhexie.com

import numpy as np
import os
import torch

import utils.io
import utils.data_transforms

from tqdm import tqdm


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
        self.cfg = cfg
        self.fields = [
            {"name": "hf", "callback": self._get_height_field},
            {"name": "ctr", "callback": self._get_footprint_contour},
            {"name": "seg", "callback": self._get_seg_map},
        ]
        self.split = split
        self.cities = self._get_cities(cfg, split)
        self.n_cities = len(self.cities)
        self.transforms = self._get_data_transforms(cfg, split)

    def __len__(self):
        return (
            self.n_cities * self.cfg.DATASETS.OSM_LAYOUT.N_REPEAT
            if self.split == "train"
            else self.n_cities
        )

    def __getitem__(self, idx):
        city = self.cities[idx % self.n_cities]
        data = {}
        for f in self.fields:
            fn = f["name"]
            if fn in self.cfg.DATASETS.OSM_LAYOUT.PIN_MEMORY:
                data[fn] = city[fn]
            else:
                data[fn] = f["callback"](city[fn])

        img = self.transforms(data)
        img = torch.cat([data[f["name"]] for f in self.fields], dim=0)
        return {"input": img, "output": img}

    def _get_cities(self, cfg, split):
        cities = sorted(os.listdir(cfg.DATASETS.OSM_LAYOUT.DIR))
        cities = cities[:-1] if split == "train" else cities[-1:]
        files = [
            {
                "hf": os.path.join(cfg.DATASETS.OSM_LAYOUT.DIR, c, "hf.png"),
                "ctr": os.path.join(cfg.DATASETS.OSM_LAYOUT.DIR, c, "ctr.png"),
                "seg": os.path.join(cfg.DATASETS.OSM_LAYOUT.DIR, c, "seg.png"),
            }
            for c in cities
        ]
        if not cfg.DATASETS.OSM_LAYOUT.PIN_MEMORY:
            return files

        return [
            {
                "hf": self._get_height_field(f["hf"])
                if "hf" in cfg.DATASETS.OSM_LAYOUT.PIN_MEMORY
                else f["hf"],
                "ctr": self._get_footprint_contour(f["ctr"])
                if "ctr" in cfg.DATASETS.OSM_LAYOUT.PIN_MEMORY
                else f["ctr"],
                "seg": self._get_seg_map(f["seg"])
                if "seg" in cfg.DATASETS.OSM_LAYOUT.PIN_MEMORY
                else f["seg"],
            }
            for f in tqdm(files, desc="Loading OSMLayout to RAM")
        ]

    def _get_height_field(self, hf_file_path):
        if utils.io.IO.get(hf_file_path) is None:
            import logging

            logging.error(hf_file_path)
        return (
            np.array(utils.io.IO.get(hf_file_path))
            / self.cfg.DATASETS.OSM_LAYOUT.MAX_HEIGHT
        )

    def _get_footprint_contour(self, footprint_ctr_file_path):
        return np.array(utils.io.IO.get(footprint_ctr_file_path).convert("L")) / 255.0

    def _get_seg_map(self, seg_map_file_path):
        return np.array(utils.io.IO.get(seg_map_file_path).convert("P"))

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
                        "objects": ["hf", "ctr", "seg"],
                    },
                    {
                        "callback": "RandomFlip",
                        "parameters": None,
                        "objects": ["hf", "ctr", "seg"],
                    },
                    {
                        "callback": "ToOneHot",
                        "parameters": {
                            "n_classes": cfg.DATASETS.OSM_LAYOUT.N_CLASSES,
                            "ignored_classes": cfg.DATASETS.OSM_LAYOUT.IGNORED_CLASSES,
                        },
                        "objects": ["seg"],
                    },
                    {
                        "callback": "ToTensor",
                        "parameters": None,
                        "objects": ["hf", "ctr", "seg"],
                    },
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
                        "objects": ["hf", "ctr", "seg"],
                    },
                    {
                        "callback": "ToOneHot",
                        "parameters": {
                            "n_classes": cfg.DATASETS.OSM_LAYOUT.N_CLASSES,
                            "ignored_classes": cfg.DATASETS.OSM_LAYOUT.IGNORED_CLASSES,
                        },
                        "objects": ["seg"],
                    },
                    {
                        "callback": "ToTensor",
                        "parameters": None,
                        "objects": ["hf", "ctr", "seg"],
                    },
                ]
            )
