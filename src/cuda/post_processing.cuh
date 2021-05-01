#ifndef JETRACER_POST_PROCESSING_H
#define JETRACER_POST_PROCESSING_H

#include <iostream>
#include <string_view>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <eigen3/Eigen/Eigen>
#include "../RealSense/RealSenseD400.h"

#include "../SlamGpuPipeline/types.h"

namespace Jetracer
{
    void overlay_keypoints(unsigned char *d_image,
                           std::size_t pitch,
                           float2 *d_pos,
                           unsigned int *d_aligned_depth,
                           int keypoints_num,
                           const rs2_intrinsics *d_rgb_intrin,
                           cudaStream_t stream);

    // void match_keypoints(float2 *d_pos_tmp,
    //                      double *d_pos,
    //                      uint32_t *d_descriptors_prev,
    //                      uint32_t *d_descriptors_curr,
    //                      double *d_points_prev,
    //                      int max_pixel_distance,
    //                      int max_hamming_distance,
    //                      int keypoints_num_prev,
    //                      int keypoints_num_curr,
    //                      int *d_keypoints_num_matched,
    //                      Eigen::Matrix4d T_w2c_prev_curr,
    //                      const rs2_intrinsics *d_rgb_intrin,
    //                      cudaStream_t stream);

    void match_keypoints(std::shared_ptr<slam_frame_t> current_frame,
                         std::shared_ptr<slam_frame_t> previous_frame,
                         int max_pixel_distance,
                         int max_hamming_distance,
                         Eigen::Matrix4d T_w2c_prev_curr,
                         int *d_valid_keypoints_num,
                         int *d_keypoints_num_matched,
                         int *h_keypoints_num_matched,
                         double3 *h_current_matched_points,
                         double3 *h_previous_matched_points,
                         const rs2_intrinsics *d_rgb_intrin,
                         cudaStream_t stream);

} // namespace Jetracer

#endif // JETRACER_POST_PROCESSING_H