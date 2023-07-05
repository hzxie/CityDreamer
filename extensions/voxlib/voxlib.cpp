// Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, check out LICENSE.md
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>

// Fast voxel traversal along rays
std::vector<torch::Tensor> ray_voxel_intersection_perspective_cuda(
    const torch::Tensor &in_voxel, const torch::Tensor &cam_ori,
    const torch::Tensor &cam_dir, const torch::Tensor &cam_up, float cam_f,
    const std::vector<float> &cam_c, const std::vector<int> &img_dims,
    int max_samples);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ray_voxel_intersection_perspective",
        &ray_voxel_intersection_perspective_cuda,
        "Ray-voxel intersections given perspective camera parameters (CUDA)");
}
