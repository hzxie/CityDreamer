# -*- coding: utf-8 -*-
#
# @File:   perceptual.py
# @Author: NVIDIA CORPORATION & AFFILIATES
# @Date:   2023-05-10 20:08:17
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-05-11 15:31:08
# @Email:  root@haozhexie.com
# @Ref: https://github.com/NVlabs/imaginaire

import torch
import torch.nn.functional as F
import torchvision

import utils.distributed


class PerceptualLoss(torch.nn.Module):
    r"""Perceptual loss initialization.

    Args:
       network (str) : The name of the loss network: 'vgg16' | 'vgg19'.
       layers (str or list of str) : The layers used to compute the loss.
       weights (float or list of float : The loss weights of each layer.
       criterion (str): The type of distance function: 'l1' | 'l2'.
       resize (bool) : If ``True``, resize the input images to 224x224.
       resize_mode (str): Algorithm used for resizing.
       num_scales (int): The loss will be evaluated at original size and
        this many times downsampled sizes.
       per_sample_weight (bool): Output loss for individual samples in the
        batch instead of mean loss.
    """

    def __init__(
        self,
        network="vgg19",
        layers="relu_4_1",
        weights=None,
        criterion="l1",
        resize=False,
        resize_mode="bilinear",
        num_scales=1,
        per_sample_weight=False,
        device="cpu",
    ):
        super().__init__()
        if isinstance(layers, str):
            layers = [layers]
        if weights is None:
            weights = [1.0] * len(layers)
        elif isinstance(layers, float) or isinstance(layers, int):
            weights = [weights]

        assert len(layers) == len(weights), (
            "The number of layers (%s) must be equal to "
            "the number of weights (%s)." % (len(layers), len(weights))
        )
        if network == "vgg19":
            self.model = vgg19(layers).to(device)
        elif network == "vgg16":
            self.model = vgg16(layers).to(device)
        else:
            raise ValueError("Network %s is not recognized" % network)

        self.num_scales = num_scales
        self.layers = layers
        self.weights = weights
        self.resize = resize
        self.resize_mode = resize_mode
        reduction = "mean" if not per_sample_weight else "none"
        if criterion == "l1":
            self.criterion = torch.nn.L1Loss(reduction=reduction)
        elif criterion == "l2" or criterion == "mse":
            self.criterion = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError("Criterion %s is not recognized" % criterion)

    def _normalize(self, input):
        r"""Normalize using ImageNet mean and std.

        Args:
            input (4D tensor NxCxHxW): The input images, assuming to be [-1, 1].

        Returns:
            Normalized inputs using the ImageNet normalization.
        """
        # normalize the input back to [0, 1]
        normalized_input = (input + 1) / 2
        # normalize the input using the ImageNet mean and std
        mean = normalized_input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = normalized_input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        output = (normalized_input - mean) / std
        return output

    def forward(self, inp, target, per_sample_weights=None):
        r"""Perceptual loss forward.

        Args:
           inp (4D tensor) : Input tensor.
           target (4D tensor) : Ground truth tensor, same shape as the input.
           per_sample_weight (bool): Output loss for individual samples in the
            batch instead of mean loss.
        Returns:
           (scalar tensor) : The perceptual loss.
        """
        # Perceptual loss should operate in eval mode by default.
        self.model.eval()
        inp, target = self._normalize(inp), self._normalize(target)
        if self.resize:
            inp = F.interpolate(
                inp, mode=self.resize_mode, size=(224, 224), align_corners=False
            )
            target = F.interpolate(
                target, mode=self.resize_mode, size=(224, 224), align_corners=False
            )

        # Evaluate perceptual loss at each scale.
        loss = 0
        for scale in range(self.num_scales):
            input_features, target_features = self.model(inp), self.model(target)

            for layer, weight in zip(self.layers, self.weights):
                l_tmp = self.criterion(
                    input_features[layer], target_features[layer].detach()
                )
                if per_sample_weights is not None:
                    l_tmp = l_tmp.mean(1).mean(1).mean(1)
                loss += weight * l_tmp
            # Downsample the input and target.
            if scale != self.num_scales - 1:
                inp = F.interpolate(
                    inp,
                    mode=self.resize_mode,
                    scale_factor=0.5,
                    align_corners=False,
                    recompute_scale_factor=True,
                )
                target = F.interpolate(
                    target,
                    mode=self.resize_mode,
                    scale_factor=0.5,
                    align_corners=False,
                    recompute_scale_factor=True,
                )

        return loss.float()


class PerceptualNetwork(torch.nn.Module):
    r"""The network that extracts features to compute the perceptual loss.

    Args:
        network (nn.Sequential) : The network that extracts features.
        layer_name_mapping (dict) : The dictionary that
            maps a layer's index to its name.
        layers (list of str): The list of layer names that we are using.
    """

    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        assert isinstance(
            network, torch.nn.Sequential
        ), 'The network needs to be of type "nn.Sequential".'
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        r"""Extract perceptual features."""
        output = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                # If the current layer is used by the perceptual loss.
                output[layer_name] = x
        return output


def vgg19(layers):
    r"""Get vgg19 layers"""
    vgg = torchvision.models.vgg19(
        weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1
    )
    # network = vgg.features
    network = torch.nn.Sequential(
        *(
            list(vgg.features)
            + [vgg.avgpool]
            + [torch.nn.Flatten()]
            + list(vgg.classifier)
        )
    )
    layer_name_mapping = {
        1: "relu_1_1",
        3: "relu_1_2",
        6: "relu_2_1",
        8: "relu_2_2",
        11: "relu_3_1",
        13: "relu_3_2",
        15: "relu_3_3",
        17: "relu_3_4",
        20: "relu_4_1",
        22: "relu_4_2",
        24: "relu_4_3",
        26: "relu_4_4",
        29: "relu_5_1",
        31: "relu_5_2",
        33: "relu_5_3",
        35: "relu_5_4",
        36: "pool_5",
        42: "fc_2",
    }
    return PerceptualNetwork(network, layer_name_mapping, layers)


def vgg16(layers):
    r"""Get vgg16 layers"""
    network = torchvision.models.vgg16(
        weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
    ).features
    layer_name_mapping = {
        1: "relu_1_1",
        3: "relu_1_2",
        6: "relu_2_1",
        8: "relu_2_2",
        11: "relu_3_1",
        13: "relu_3_2",
        15: "relu_3_3",
        18: "relu_4_1",
        20: "relu_4_2",
        22: "relu_4_3",
        25: "relu_5_1",
    }
    return PerceptualNetwork(network, layer_name_mapping, layers)
