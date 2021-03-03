#ifndef JETRACER_CUDA_ALIGN_UTILS_H
#define JETRACER_CUDA_ALIGN_UTILS_H

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <cuda_runtime.h>
#include <helper_cuda.h>

int calc_block_size(int pixel_count, int thread_count);

__device__ void kernel_transfer_pixels(int2 *mapped_pixels,
                                       const rs2_intrinsics *depth_intrin,
                                       const rs2_intrinsics *other_intrin,
                                       const rs2_extrinsics *depth_to_other,
                                       float depth_val,
                                       int depth_x,
                                       int depth_y,
                                       int block_index);

__global__ void kernel_map_depth_to_other(int2 *mapped_pixels,
                                          const uint16_t *depth_in,
                                          const rs2_intrinsics *depth_intrin,
                                          const rs2_intrinsics *other_intrin,
                                          const rs2_extrinsics *depth_to_other,
                                          float depth_scale);

__global__ void kernel_depth_to_other(uint16_t *aligned_out,
                                      const uint16_t *depth_in,
                                      const int2 *mapped_pixels,
                                      const rs2_intrinsics *depth_intrin,
                                      const rs2_intrinsics *other_intrin);

__global__ void kernel_replace_to_zero(uint16_t *aligned_out,
                                       const rs2_intrinsics *other_intrin);

#endif // JETRACER_CUDA_ALIGN_UTILS_H
