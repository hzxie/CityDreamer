# -*- coding: utf-8 -*-
#
# @File:   footage_segmentation.py
# @Author: Haozhe Xie
# @Date:   2023-05-01 10:27:01
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-05-02 20:17:32
# @Email:  root@haozhexie.com
#
# Quick Start
# - conda create --name seem python=3.8
# - conda activate seem
# - git clone https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git
# - mv Segment-Everything-Everywhere-All-At-Once SEEM
# - cd SEEM/demo_code
# - pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# - pip install -r requirements.txt
#
# Important Note
# - MAKE SURE THE SCRIPT IS RUNNING IN THE ``seem'' ENVIRONMENT.
#
# Reference
# - https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once

import argparse
import detectron2.data
import importlib
import logging
import math
import numpy as np
import os
import sys
import torch
import torchvision.transforms
import urllib.request

from PIL import Image
from tqdm import tqdm


def get_seem_model(seem_home, seem_cfg):
    arguments = importlib.import_module("utils.arguments")
    base_model = importlib.import_module("xdecoder.BaseModel")
    constants = importlib.import_module("utils.constants")
    distributed = importlib.import_module("utils.distributed")
    xdecoder = importlib.import_module("xdecoder")
    opt = arguments.load_opt_from_config_files(os.path.join(seem_home, seem_cfg))
    opt = distributed.init_distributed(opt)
    ckpt_file_path = os.path.join(seem_home, "seem_focalt_v1.pt")
    if not os.path.exists(ckpt_file_path):
        urllib.request.urlretrieve(
            "https://projects4jw.blob.core.windows.net/x-decoder/release/seem_focalt_v1.pt",
            ckpt_file_path,
        )

    model = (
        base_model.BaseModel(opt, xdecoder.build_model(opt))
        .from_pretrained(ckpt_file_path)
        .eval()
        .cuda()
    )
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            constants.COCO_PANOPTIC_CLASSES + ["background"], is_eval=True
        )
    # Set up classes
    classes = [
        name.replace("-other", "").replace("-merged", "")
        for name in constants.COCO_PANOPTIC_CLASSES
    ] + ["others"]
    model.model.metadata = detectron2.data.MetadataCatalog.get(
        "coco_2017_train_panoptic"
    )
    return model, classes


def get_transformed_image(image, transformer):
    return {
        "image": torch.from_numpy(np.asarray(transformer(image)))
        .permute(2, 0, 1)
        .cuda(),
        "height": image.size[1],
        "width": image.size[0],
    }


def get_seg_map(seg_map, seg_map_classes, all_classes, unknown_classes):
    # CG Class: [None, Road, Building, Tree, Construction, Water, Other]
    CLASS_MAPPER = {
        "road": 1,
        "railroad": 1,
        "bridge": 1,
        "building": 2,
        "house": 2,
        "roof": 2,
        "tree": 3,
        "grass": 3,
        "sea": 5,
        "river": 5,
        "water": 5,
        "dirt": 6,
        "pavement": 6,
        "others": 6,
    }
    # Use 255 as default values
    cg_seg_map = np.ones_like(seg_map) * 255
    for smc in seg_map_classes:
        cat_id = smc["id"]
        cat_name = all_classes[smc["category_id"]]
        if cat_name not in CLASS_MAPPER:
            if cat_name not in unknown_classes:
                unknown_classes.add(cat_name)
                logging.warning("Class[Name=%s] is ignored." % cat_name)
            continue
        new_cat_id = CLASS_MAPPER[cat_name]
        cg_seg_map[seg_map == cat_id] = new_cat_id

    # Generate the palette
    PALETTE = np.array([[i, i, i] for i in range(256)])
    # fmt: off
    PALETTE[:7] = np.array(
        [
            [0, 0, 0],       # empty        -> black (ONLY used in voxel)
            [96, 0, 0],      # road         -> red
            [96, 96, 0],     # building     -> yellow
            [0, 96, 0],      # green lands  -> green
            [0, 96, 96],     # construction -> cyan
            [0, 0, 96],      # water        -> blue
            [128, 128, 128], # ground       -> gray
        ]
    )
    # fmt: on
    cg_seg_map = Image.fromarray(cg_seg_map.astype(np.uint8))
    cg_seg_map.putpalette(PALETTE.reshape(-1).tolist())
    return cg_seg_map


def get_frames_with_seg_map(frame, seg_map):
    frame = np.array(frame)
    seg_map = np.array(seg_map.convert("RGB"))
    return Image.fromarray((frame * 0.6 + seg_map * 0.4).astype(np.uint8))


def main(seem_home, seem_cfg, ges_dir, output_dir, batch_size, debug):
    sys.path.append(os.path.join(seem_home))
    # Set up SEEM model
    model, all_classes = get_seem_model(seem_home, seem_cfg)
    # Set up image transformation
    transformer = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                512,
                interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC,
            )
        ]
    )
    # Segmentation
    unknown_classes = set()
    ge_projects = sorted(os.listdir(ges_dir))
    for gep in tqdm(ge_projects):
        files = sorted(os.listdir(os.path.join(ges_dir, gep, "footage")))
        frames = [Image.open(os.path.join(ges_dir, gep, "footage", f)) for f in files]
        tr_frames = [get_transformed_image(f, transformer) for f in frames]
        n_batches = math.ceil(len(frames) / batch_size)

        # Skip folders that all seg maps are generated
        _output_dir = os.path.join(ges_dir, gep, output_dir)
        if os.path.exists(_output_dir):
            are_all_files_generated = True
            expected_files = ["%s.png" % os.path.splitext(fn)[1] for fn in files]
            for ef in expected_files:
                if not os.path.exists(os.path.join(ef)):
                    are_all_files_generated = False
            if are_all_files_generated:
                continue

        os.makedirs(_output_dir, exist_ok=True)
        for i in range(n_batches):
            s_idx = batch_size * i
            e_idx = s_idx + batch_size
            _files = files[s_idx:e_idx]
            _frames = frames[s_idx:e_idx]
            _tr_frames = tr_frames[s_idx:e_idx]
            with torch.no_grad():
                _results = model.model.evaluate(_tr_frames)

            torch.cuda.empty_cache()
            for fr, r, fn in zip(_frames, _results, _files):
                seg_map = get_seg_map(
                    r["panoptic_seg"][0].cpu().numpy(),
                    r["panoptic_seg"][1],
                    all_classes,
                    unknown_classes,
                )
                basename, _ = os.path.splitext(fn)
                seg_map.save(os.path.join(_output_dir, "%s.png" % basename))
                if debug:
                    get_frames_with_seg_map(fr, seg_map).save(
                        os.path.join(_output_dir, "%s.jpg" % basename)
                    )


if __name__ == "__main__":
    PROJECT_HOME = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir)
    )
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.WARNING,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seem_home",
        help="The path to the project dir of SEEM",
        default=os.path.join(PROJECT_HOME, os.pardir, "SEEM", "demo_code"),
    )
    parser.add_argument(
        "--seem_cfg",
        help="The path to the config file of SEEM",
        default=os.path.join("configs", "seem", "seem_focalt_lang.yaml"),
    )
    parser.add_argument("--ges_dir", default=os.path.join(PROJECT_HOME, "data", "ges"))
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--output_dir", default="seg")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(
        args.seem_home,
        args.seem_cfg,
        args.ges_dir,
        args.output_dir,
        args.batch_size,
        args.debug,
    )
