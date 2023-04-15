/**
 * @File:   extrude_tensor_ext.cu
 * @Author: Haozhe Xie
 * @Date:   2023-03-26 11:06:18
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2023-04-15 13:27:39
 * @Email:  root@haozhexie.com
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <torch/extension.h>

#define CUDA_NUM_THREADS 512

// Computer the number of threads needed in GPU
inline int get_n_threads(int n) {
  const int pow_2 = std::log(static_cast<float>(n)) / std::log(2.0);
  return max(min(1 << pow_2, CUDA_NUM_THREADS), 1);
}

__global__ void extrude_tensor_ext_cuda_kernel(
    int height, int width, int depth, const int *__restrict__ seg_map,
    const int *__restrict__ height_field, int *__restrict__ volume) {
  int batch_index = blockIdx.x;
  int index = threadIdx.x;
  int stride = blockDim.x;

  seg_map += batch_index * height * width;
  height_field += batch_index * height * width;
  volume += batch_index * height * width * depth;
  for (int i = index; i < height; i += stride) {
    int offset_2d_r = i * width, offset_3d_r = i * width * depth;
    for (int j = 0; j < width; ++j) {
      int offset_2d_c = offset_2d_r + j, offset_3d_c = offset_3d_r + j * depth;
      int seg = seg_map[offset_2d_c];
      int hf = height_field[offset_2d_c];
      for (int k = 0; k < hf + 1; ++k) {
        volume[offset_3d_c + k] = seg;
      }
    }
  }
}

torch::Tensor extrude_tensor_ext_cuda_forward(torch::Tensor seg_map,
                                              torch::Tensor height_field,
                                              int max_height,
                                              cudaStream_t stream) {
  int batch_size = seg_map.size(0);
  int height = seg_map.size(2);
  int width = seg_map.size(3);
  torch::Tensor volume = torch::zeros({batch_size, height, width, max_height},
                                      torch::CUDA(torch::kInt));

  extrude_tensor_ext_cuda_kernel<<<
      batch_size, int(CUDA_NUM_THREADS / CUDA_NUM_THREADS), 0, stream>>>(
      height, width, max_height, seg_map.data_ptr<int>(),
      height_field.data_ptr<int>(), volume.data_ptr<int>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in extrude_tensor_ext_cuda_forward: %s\n", cudaGetErrorString(err));
  }
  return volume;
}
