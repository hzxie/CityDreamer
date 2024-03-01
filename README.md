
<img src="https://www.infinitescript.com/projects/CityDreamer/CityDreamer-Logo.png" height="150px" align="right">

# CityDreamer: Compositional Generative Model of Unbounded 3D Cities

[Haozhe Xie](https://haozhexie.com), [Zhaoxi Chen](https://frozenburning.github.io/), [Fangzhou Hong](https://hongfz16.github.io/), [Ziwei Liu](https://liuziwei7.github.io/)

S-Lab, Nanyang Technological University

![Counter](https://api.infinitescript.com/badgen/count?name=hzxie/CityDreamer)
[![arXiv](https://img.shields.io/badge/arXiv-2309.00610-b31b1b.svg)](https://arxiv.org/abs/2309.00610)
[![YouTube](https://img.shields.io/badge/Spotlight%20Video-%23FF0000.svg?logo=YouTube&logoColor=white)](https://youtu.be/te4zinLTYz0)

![Teaser](https://www.infinitescript.com/projects/CityDreamer/CityDreamer-Teaser.jpg)

## Update🔥

- [2023/03/28] The training and testing code is released.
- [2023/02/27] The OSM and GoogleEarth datasets is released.
- [2023/08/15] The repo is created.

## Cite this work📝

```
@inproceedings{xie2024citydreamer,
  title     = {City{D}reamer: Compositional Generative Model of Unbounded 3{D} Cities},
  author    = {Xie, Haozhe and 
               Chen, Zhaoxi and 
               Hong, Fangzhou and 
               Liu, Ziwei},
  booktitle = {CVPR},
  year      = {2024}
}
```

## Datasets and Pretrained Models 🛢️

The proposed OSM and GoogleEarth datasets  are available as below.

- [OSM](https://gateway.infinitescript.com/s/OSM)
- [GoogleEarth](https://gateway.infinitescript.com/s/GoogleEarth)

The pretrained models are available as below.

- [Unbounded Layout Generator](https://gateway.infinitescript.com/s/LayoutGen.pth)
- [Background Stuff Generator](https://gateway.infinitescript.com/s/CityDreamer-Bgnd.pth)
- [Building Instance Generator](https://gateway.infinitescript.com/s/CityDreamer-Fgnd.pth)

## Installation📥

Assume that you have installed [CUDA](https://developer.nvidia.com/cuda-downloads) and [PyTorch](https://pytorch.org) in your Python (or Anaconda) environment.  

The CityDreamer source code is tested in PyTorch 1.13.1 with CUDA 11.7. You can use the following command to install PyTorch with CUDA 11.7.

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

After that, the Python dependencies can be installed as following.

```bash
git clone https://github.com/hzxie/city-dreamer
cd city-dreamer
CITY_DREAMER_HOME=`pwd`
pip install -r requirements.txt
```

The CUDA extensions can be compiled and installed with the following commands.

```bash
cd $CITY_DREAMER_HOME/extensions
for e in `ls -d */`
do
  cd $CITY_DREAMER_HOME/extensions/$e
  pip install .
done
```

## Inference🚩

Both the iterative demo and command line interface (CLI) by default load the pretrained models for Unbounded Layout Generator, Background Stuff Generator, and Building Instance Generator from `output/sampler.pth`, `output/gancraft-bg.pth`, and `output/gancraft-fg.pth`, respectively. You have the option to specify a different location using runtime arguments.

```
├── ...
└── city-dreamer
    └── demo
    |   ├── ...
    |   └── run.py
    └── scripts
    |   ├── ...
    |   └── inference.py
    └── output
        ├── gancraft-bg.pth
        ├── gancraft-fg.pth
        └── sampler.pth
```

Moreover, both scripts feature runtime arguments `--patch_height` and `--patch_width`, which divide images into patches of size `patch_height`x`patch_width`. For a single NVIDIA RTX 3090 GPU with 24GB of VRAM, both patch_height and patch_width are set to 5. You can adjust the values to match your GPU's VRAM size.

### Iterative Demo 🕹️

```bash
python3 demo/run.py
```

Then, open http://localhost:3186 in your browser.

### Command Line Interface (CLI) 🤖

```bash
python3 scripts/inference.py
```

The generated video is located at `output/rendering.mp4`.

## Training👩🏽‍💻

### Dataset Preparation

By default, all scripts load the [OSM](https://gateway.infinitescript.com/s/OSM) and [GoogleEarth](https://gateway.infinitescript.com/s/GoogleEarth) datasets from `./data/osm` and `./data/ges`, respectively. You have the option to specify a different location using runtime arguments.

```
├── ...
└── city-dreamer
    └── data
        ├── ges  # GoogleEarth
        └── osm  # OSM
```

The instance segmentation annotation for the GoogleEarth dataset needs to be generated as following steps (requiring approximately 1TB of disk space).

1. Generate semantic segmentation using [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once).

```bash
git clone -b v1.0 https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git
mv Segment-Everything-Everywhere-All-At-Once $CITY_DREAMER_HOME/../SEEM
cd $CITY_DREAMER_HOME/../SEEM
# Remove the PyTorch 2.1.0 dependency. PyTorch 1.13.1 is also OK for SEEM.
sed -i "/torch/d" assets/requirements/requirements.txt
# Install the dependencies for SEEM
pip install -r assets/requirements/requirements.txt
pip install -r assets/requirements/requirements_custom.txt
# Back to the CityDreamer codebase
cd $CITY_DREAMER_HOME
python3 scripts/footage_segmentation.py
```

2. Generate instance segmetation.

```bash
cd $CITY_DREAMER_HOME
python3 scripts/dataset_generator.py
```

### Unbounded Layout Generator Training

Unbounded Layout Generator consists of two networks: `VQVAE` and `Sampler`.

#### Launch Training 🚀

```bash
# 0x01. Train VQVAE with 4 GPUs
torchrun --nnodes=1 --nproc_per_node=4 --standalone run.py -n VQGAN -e VQGAN-Exp

# 0x02. Train Sampler with 2 GPUs
torchrun --nnodes=1 --nproc_per_node=2 --standalone run.py -n Sampler -e Sampler-Exp\
         -p /output/checkpoints/VQGAN-Exp/ckpt-last.pth
```

### Background Stuff Generator Training

#### Update `config.py` ⚙️

Make sure the config matches the following lines.

```python
cfg.NETWORK.GANCRAFT.BUILDING_MODE               = False
cfg.TRAIN.GANCRAFT.REC_LOSS_FACTOR               = 10
cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_FACTOR        = 10
cfg.TRAIN.GANCRAFT.GAN_LOSS_FACTOR               = 0.5
```

#### Launch Training 🚀

```bash
# 0x03. Train Background Stuff Generator with 8 GPUs
torchrun --nnodes=1 --nproc_per_node=8 --standalone run.py -n GANCraft -e BSG-Exp
```

### Building Instance Generator Training

#### Update `config.py` ⚙️

Make sure the config matches the following lines.

```python
cfg.NETWORK.GANCRAFT.BUILDING_MODE               = True
cfg.TRAIN.GANCRAFT.REC_LOSS_FACTOR               = 0
cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_FACTOR        = 0
cfg.TRAIN.GANCRAFT.GAN_LOSS_FACTOR               = 1
```

#### Launch Training 🚀

```bash
# 0x04. Train Building Instance Generator with 8 GPUs
torchrun --nnodes=1 --nproc_per_node=8 --standalone run.py -n GANCraft -e BIG-Exp
```

## License

This project is licensed under [NTU S-Lab License 1.0](https://github.com/hzxie/city-dreamer/blob/master/LICENSE). Redistribution and use should follow this license.
