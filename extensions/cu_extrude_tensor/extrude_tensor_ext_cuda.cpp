/**
 * @File:   extrude_tensor_ext_cuda.cpp
 * @Author: Haozhe Xie
 * @Date:   2023-03-26 11:06:13
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2023-03-26 16:28:20
 * @Email:  root@haozhexie.com
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

torch::Tensor extrude_tensor_ext_cuda_forward(torch::Tensor seg_map,
                                              torch::Tensor height_field,
                                              int max_height,
                                              cudaStream_t stream);

torch::Tensor extrude_tensor_ext_forward(torch::Tensor seg_map,
                                         torch::Tensor height_field,
                                         int max_height) {
  CHECK_INPUT(seg_map);
  CHECK_INPUT(height_field);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return extrude_tensor_ext_cuda_forward(seg_map, height_field, max_height,
                                         stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &extrude_tensor_ext_forward,
        "Extrude Tensor Ext. Forward (CUDA)");
}
