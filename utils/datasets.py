# -*- coding: utf-8 -*-
#
# @File:   data_loaders.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 10:29:53
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-05-06 14:36:51
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
    elif dataset_name == "GOOGLE_EARTH":
        return GoogleEarthDataset(cfg, split)
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
            {"name": "hf", "callback": self.get_height_field},
            {"name": "ctr", "callback": self.get_footprint_contour},
            {"name": "seg", "callback": self.get_seg_map},
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
                data[fn] = f["callback"](city[fn], self.cfg)

        img = self.transforms(data)
        img = torch.cat([data[f["name"]] for f in self.fields], dim=0)
        return {"img": img}

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
                fld["name"]: fld["callback"](f[fld["name"]], cfg)
                if fld["name"] in cfg.DATASETS.OSM_LAYOUT.PIN_MEMORY
                else f[fld["name"]]
                for fld in self.fields
            }
            for f in tqdm(files, desc="Loading OSMLayout to RAM")
        ]

    @classmethod
    def get_height_field(self, hf_file_path, cfg):
        return (
            np.array(utils.io.IO.get(hf_file_path)) / cfg.DATASETS.OSM_LAYOUT.MAX_HEIGHT
        )

    @classmethod
    def get_footprint_contour(self, footprint_ctr_file_path, _=None):
        return np.array(utils.io.IO.get(footprint_ctr_file_path).convert("L")) / 255.0

    @classmethod
    def get_seg_map(self, seg_map_file_path, _=None):
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


class GoogleEarthDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.fields = ["hf", "seg", "footage", "raycasting"]
        self.memcached = {}
        self.trajectories = self._get_trajectories(cfg, split)
        self.n_trajectories = len(self.trajectories)
        self.transforms = self._get_data_transforms(cfg, split)

    def __len__(self):
        return (
            self.n_trajectories * self.cfg.DATASETS.GOOGLE_EARTH.N_REPEAT
            if self.split == "train"
            else self.n_trajectories
        )

    def __getitem__(self, idx):
        trajectory = self.trajectories[idx % self.n_trajectories]

        data = {"footage": self._get_footage_img(trajectory["footage"])}
        raycasting = utils.io.IO.get(trajectory["raycasting"])
        data["voxel_id"] = raycasting["voxel_id"]
        data["depth2"] = raycasting["depth2"]
        data["raydirs"] = raycasting["raydirs"]
        data["cam_ori_t"] = raycasting["cam_ori_t"]
        data["mask"] = raycasting["mask"]
        data["hf"] = self._get_hf_seg(
            "hf",
            trajectory,
            raycasting["img_center"]["cx"],
            raycasting["img_center"]["cy"],
        )
        data["seg"] = self._get_hf_seg(
            "seg",
            trajectory,
            raycasting["img_center"]["cx"],
            raycasting["img_center"]["cy"],
        )
        data = self.transforms(data)
        return data

    def _get_hf_seg(self, field, trajectory, cx, cy):
        if field in self.cfg.DATASETS.GOOGLE_EARTH.PIN_MEMORY:
            img = self.memcached[trajectory[field]]
        elif field == "hf":
            img = OsmLayoutDataset.get_height_field(trajectory[field], self.cfg)
        elif field == "seg":
            img = OsmLayoutDataset.get_seg_map(trajectory[field])
        else:
            raise Exception("Unknown field: %s" % field)

        half_size = self.cfg.DATASETS.GOOGLE_EARTH.VOL_SIZE // 2
        tl_x, br_x = cx - half_size, cx + half_size
        tl_y, br_y = cy - half_size, cy + half_size
        return img[tl_y:br_y, tl_x:br_x]

    def _get_trajectory_city(self, trajectory):
        # Trajectory name example: US-SanFrancisco-Chinatown-R624-A354
        return "-".join(trajectory.split("-")[:2])

    def _get_trajectories(self, cfg, split):
        trajectories = sorted(os.listdir(cfg.DATASETS.GOOGLE_EARTH.DIR))
        trajectories = trajectories[:-1] if split == "train" else trajectories[-1:]
        trajectories = ["US-NewYork-1NewYorkPlaza-R813-A884"]
        files = [
            {
                "hf": os.path.join(
                    cfg.DATASETS.OSM_LAYOUT.DIR, self._get_trajectory_city(t), "hf.png"
                ),
                "seg": os.path.join(
                    cfg.DATASETS.OSM_LAYOUT.DIR, self._get_trajectory_city(t), "seg.png"
                ),
                "footage": os.path.join(
                    cfg.DATASETS.GOOGLE_EARTH.DIR, t, "footage", "%s_%02d.jpeg" % (t, i)
                ),
                "raycasting": os.path.join(
                    cfg.DATASETS.GOOGLE_EARTH.DIR,
                    t,
                    "raycasting",
                    "%s_%02d.pkl" % (t, i),
                ),
            }
            for t in trajectories
            for i in range(cfg.DATASETS.GOOGLE_EARTH.N_VIEWS)
        ]
        if not cfg.DATASETS.GOOGLE_EARTH.PIN_MEMORY:
            return files

        for f in tqdm(files, desc="Loading partial files to RAM"):
            for k, v in f.items():
                if k not in cfg.DATASETS.GOOGLE_EARTH.PIN_MEMORY:
                    continue
                elif v in self.memcached:
                    continue
                elif k == "hf":
                    self.memcached[v] = OsmLayoutDataset.get_height_field(v, cfg)
                elif k == "seg":
                    self.memcached[v] = OsmLayoutDataset.get_seg_map(v)

        return files

    def _get_footage_img(self, img_file_path):
        img = utils.io.IO.get(img_file_path)
        return (np.array(img) / 255.0 - 0.5) * 2

    def _get_data_transforms(self, cfg, split):
        if split == "train":
            return utils.data_transforms.Compose(
                [
                    {
                        "callback": "RandomCrop",
                        "parameters": {
                            "height": cfg.TRAIN.GANCRAFT.CROP_SIZE[0],
                            "width": cfg.TRAIN.GANCRAFT.CROP_SIZE[1],
                        },
                        "objects": ["voxel_id", "depth2", "raydirs", "footage", "mask"],
                    },
                    {
                        "callback": "ToOneHot",
                        "parameters": {
                            "n_classes": cfg.DATASETS.OSM_LAYOUT.N_CLASSES,
                        },
                        "objects": ["seg"],
                    },
                    {
                        "callback": "ToTensor",
                        "parameters": None,
                        "objects": [
                            "hf",
                            "seg",
                            "voxel_id",
                            "depth2",
                            "raydirs",
                            "cam_ori_t",
                            "footage",
                            "mask",
                        ],
                    },
                ]
            )
        else:
            return utils.data_transforms.Compose(
                [
                    {
                        "callback": "CenterCrop",
                        "parameters": {
                            "height": cfg.TEST.GANCRAFT.CROP_SIZE[0],
                            "width": cfg.TEST.GANCRAFT.CROP_SIZE[1],
                        },
                        "objects": ["voxel_id", "depth2", "raydirs", "footage", "mask"],
                    },
                    {
                        "callback": "ToOneHot",
                        "parameters": {
                            "n_classes": cfg.DATASETS.OSM_LAYOUT.N_CLASSES,
                        },
                        "objects": ["seg"],
                    },
                    {
                        "callback": "ToTensor",
                        "parameters": None,
                        "objects": [
                            "hf",
                            "seg",
                            "voxel_id",
                            "depth2",
                            "raydirs",
                            "cam_ori_t",
                            "footage",
                            "mask",
                        ],
                    },
                ]
            )
