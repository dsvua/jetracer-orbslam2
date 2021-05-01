#ifndef JETRACER_CUDA_ALIGN_UTILS_H
#define JETRACER_CUDA_ALIGN_UTILS_H

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace Jetracer
{
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

    __global__ void kernel_depth_to_other(unsigned int *aligned_out,
                                          const uint16_t *depth_in,
                                          const int2 *mapped_pixels,
                                          const rs2_intrinsics *depth_intrin,
                                          const rs2_intrinsics *other_intrin);

    __global__ void kernel_reset_to_zero(unsigned int *aligned_out,
                                         const rs2_intrinsics *other_intrin);

    void align_depth_to_other(unsigned int *d_aligned_out,
                              const uint16_t *d_depth_in,
                              int2 *d_pixel_map,
                              float depth_scale,
                              int image_width,
                              int image_height,
                              const rs2_intrinsics *d_depth_intrin,
                              const rs2_intrinsics *d_other_intrin,
                              const rs2_extrinsics *d_depth_to_other,
                              cudaStream_t stream);

    void keypoint_pixel_to_point(unsigned int *d_aligned_depth,
                                 const rs2_intrinsics *d_rgb_intrin,
                                 int image_width,
                                 int image_height,
                                 float2 *d_pos_out,
                                 float2 *d_pos_in,
                                 float *d_score,
                                 double *d_points,
                                 uint32_t *d_descriptors_out,
                                 uint32_t *d_descriptors_in,
                                 int keypoints_num,
                                 int *h_valid_keypoints_num,
                                 int *d_valid_keypoints_num,
                                 cudaStream_t stream);

}

#endif // JETRACER_CUDA_ALIGN_UTILS_H
