#include "post_processing.cuh"
#include "../cuda_common.h"

#include <memory>
#include <helper_cuda.h>
#include <stdio.h> //for printf

namespace Jetracer
{
    /* Given a point in 3D space, compute the corresponding pixel coordinates in an image with no distortion or forward distortion coefficients produced by the same camera */
    __device__ static void project_point_to_pixel_double(float pixel[2],
                                                         const struct rs2_intrinsics *intrin,
                                                         const double point[4])
    {
        //assert(intrin->model != RS2_DISTORTION_INVERSE_BROWN_CONRADY); // Cannot project to an inverse-distorted image

        float x = point[0] / point[2], y = point[1] / point[2];

        if (intrin->model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY)
        {

            float r2 = x * x + y * y;
            float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
            x *= f;
            y *= f;
            float dx = x + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
            float dy = y + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
            x = dx;
            y = dy;
        }

        if (intrin->model == RS2_DISTORTION_FTHETA)
        {
            float r = sqrtf(x * x + y * y);
            float rd = (float)(1.0f / intrin->coeffs[0] * atan(2 * r * tan(intrin->coeffs[0] / 2.0f)));
            x *= rd / r;
            y *= rd / r;
        }

        pixel[0] = x * intrin->fx + intrin->ppx;
        pixel[1] = y * intrin->fy + intrin->ppy;
        // printf("pixel %0.4f %0.4f \t\t\t point %0.4f %0.4f %0.4f", pixel[0], pixel[0], point[0], point[1], point[2])
    }

    __global__ void kernel_overlay_keypoints(unsigned char *d_image,
                                             std::size_t pitch,
                                             float2 *d_pos,
                                             unsigned int *d_aligned_depth,
                                             const rs2_intrinsics *d_rgb_intrin,
                                             int keypoints_num)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;

        if (idx < keypoints_num)
        {
            float2 pos = d_pos[idx];

            for (int x = pos.x - 1; x < pos.x + 1; x++)
            {
                for (int y = pos.y - 1; y < pos.y + 1; y++)
                {
                    if (y < d_rgb_intrin->height && x < d_rgb_intrin->width)
                    {
                        // if (d_aligned_depth[y * d_rgb_intrin->width + x] > 0.1f)
                        d_image[y * pitch + x] = 255;
                    }
                }
            }
        }
    }

    __global__ void kernel_reproject_prev_points(float2 *d_pos_tmp,
                                                 double *d_points_prev,
                                                 int keypoints_num_prev,
                                                 Eigen::Matrix4d *T_w2c_prev_curr_,
                                                 const rs2_intrinsics *d_rgb_intrin)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        Eigen::Matrix4d T_w2c_prev_curr = *T_w2c_prev_curr_;

        if (idx < keypoints_num_prev)
        {
            // double3 point = make_double3(d_points_prev[idx], d_points_prev[idx + 1], d_points_prev[idx + 2]);
            Eigen::Vector4d e_point(d_points_prev[idx * 3], d_points_prev[idx * 3 + 1], d_points_prev[idx * 3 + 2], 1);
            e_point = T_w2c_prev_curr * e_point;
            // printf("d_points_prev %0.4f %0.4f %0.4f\n\t\t e_point %0.4f %0.4f %0.4f\n", d_points_prev[idx * 3], d_points_prev[idx * 3 + 1], d_points_prev[idx * 3 + 2], e_point(0), e_point(1), e_point(2));

            project_point_to_pixel_double((float *)(d_pos_tmp + idx), d_rgb_intrin, (double *)e_point.data());
        }
    }

    __global__ void kernel_match_keypoints(float2 *d_pos_tmp,
                                           float2 *d_pos,
                                           uint16_t *d_pos_frame,
                                           uint32_t *d_descriptors_prev,
                                           uint32_t *d_descriptors_curr,
                                           int max_pixel_distance,
                                           int max_hamming_distance,
                                           int keypoints_num_prev,
                                           int *keypoints_num_curr,
                                           double3 *previous_matched_points,
                                           double3 *current_matched_points,
                                           double3 *previous_points,
                                           double3 *current_points,
                                           int keypoints_num,
                                           int *d_keypoints_num_matched)
    {
        __shared__ int warp_counter;
        __shared__ int global_idx;
        __shared__ uint32_t descriptors_cache[CUDA_WARP_SIZE];
        __shared__ float2 d_pos_cache[CUDA_WARP_SIZE];

        int local_idx = 0;
        int pair_idx = 0;
        bool is_matched = false;
        int hamming_distance_curr = 9999999;

        __syncthreads();

        if (threadIdx.x == 0)
        {
            warp_counter = 0;
        }

        __syncthreads();

        int idx = blockDim.x * blockIdx.x + threadIdx.x;

        if (idx < keypoints_num_prev)
        {
            uint32_t descriptor = d_descriptors_prev[idx];
            float2 pos_prev = d_pos_tmp[idx];

            for (int i = 0; i < *keypoints_num_curr; i += CUDA_WARP_SIZE)
            {
                int i_idx = i + threadIdx.x;
                if (i_idx < *keypoints_num_curr)
                {
                    descriptors_cache[threadIdx.x] = d_descriptors_curr[i_idx];
                    d_pos_cache[threadIdx.x] = d_pos[i_idx];
                };

                __syncthreads();

                int max_j_loop = CUDA_WARP_SIZE;
                if (i + CUDA_WARP_SIZE >= *keypoints_num_curr)
                {
                    max_j_loop = *keypoints_num_curr - i;
                };

                for (int j = 0; j < max_j_loop; j++)
                {
                    if (threadIdx.x < max_j_loop)
                    {
                        int j_idx = (j + threadIdx.x) % max_j_loop; // to access shared mem in parallel to avoid serialization
                        if (fabsf(pos_prev.x - d_pos_cache[j_idx].x) <= max_pixel_distance &&
                            fabsf(pos_prev.y - d_pos_cache[j_idx].y) <= max_pixel_distance)
                        {
                            int hamming_distance = __popc(descriptor ^ descriptors_cache[j_idx]);
                            // printf("pos_prev %0.4f %0.4f \t\t d_pos_cache %0.4f %0.4f \t\t hamming_distance %0.4f \t\t j_idx %d\n",
                            //     pos_prev.x, pos_prev.y, d_pos_cache[j_idx].x, d_pos_cache[j_idx].y, hamming_distance, j_idx);

                            // found pair, but still looking for better match
                            if (hamming_distance < max_hamming_distance && hamming_distance < hamming_distance_curr)
                            {
                                hamming_distance_curr = hamming_distance;
                                is_matched = true;
                                pair_idx = i + j_idx;
                            };
                        };
                    };
                };
            };

            __syncthreads();

            if (is_matched)
            {
                local_idx = atomicAdd(&warp_counter, 1);
            };

            __syncthreads();

            if (threadIdx.x == 0 && warp_counter > 0)
            {
                global_idx = atomicAdd(d_keypoints_num_matched, warp_counter);
            };

            __syncthreads();

            if (is_matched)
            {
                previous_matched_points[global_idx + local_idx] = previous_points[idx];
                current_matched_points[global_idx + local_idx] = current_points[pair_idx];
                float2 pos_tmp = d_pos[pair_idx];
                d_pos_frame[global_idx + local_idx] = uint16_t(pos_tmp.x);
                d_pos_frame[keypoints_num + global_idx + local_idx] = uint16_t(pos_tmp.y);
            };
        };
    }

    void overlay_keypoints(unsigned char *d_image,
                           std::size_t pitch,
                           float2 *d_pos,
                           unsigned int *d_aligned_depth,
                           int keypoints_num,
                           const rs2_intrinsics *d_rgb_intrin,
                           cudaStream_t stream)
    {
        dim3 threads(CUDA_WARP_SIZE);
        // int tmp_blocks = (keypoints_num % CUDA_WARP_SIZE == 0) ? keypoints_num / CUDA_WARP_SIZE : keypoints_num / CUDA_WARP_SIZE + 1;
        dim3 blocks(calc_block_size(keypoints_num, threads.x));
        kernel_overlay_keypoints<<<blocks, threads, 0, stream>>>(d_image,
                                                                 pitch,
                                                                 d_pos,
                                                                 d_aligned_depth,
                                                                 d_rgb_intrin,
                                                                 keypoints_num);
    }

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
    //                      cudaStream_t stream)
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
                         cudaStream_t stream)
    {

        Eigen::Matrix4d *T_w2c_prev_curr_;
        float2 *d_pos_tmp;
        uint16_t *d_pos_frame;
        double3 *previous_matched_points;
        double3 *current_matched_points;
        checkCudaErrors(cudaMalloc((void **)&T_w2c_prev_curr_, sizeof(Eigen::Matrix4d)));
        checkCudaErrors(cudaMalloc((void **)&d_pos_tmp, sizeof(float2) * previous_frame->keypoints_count));
        checkCudaErrors(cudaMalloc((void **)&d_pos_frame, sizeof(uint16_t) * previous_frame->keypoints_count * 2));
        checkCudaErrors(cudaMalloc((void **)&previous_matched_points, sizeof(double3) * previous_frame->keypoints_count));
        checkCudaErrors(cudaMalloc((void **)&current_matched_points, sizeof(double3) * previous_frame->keypoints_count));

        checkCudaErrors(cudaMemcpyAsync((void *)T_w2c_prev_curr_,
                                        &T_w2c_prev_curr,
                                        sizeof(Eigen::Matrix4d),
                                        cudaMemcpyHostToDevice,
                                        stream));

        checkCudaErrors(cudaMemcpyAsync((void *)d_keypoints_num_matched,
                                        h_keypoints_num_matched,
                                        sizeof(int),
                                        cudaMemcpyHostToDevice,
                                        stream));

        dim3 threads(CUDA_WARP_SIZE);
        // int tmp_blocks = (keypoints_num % CUDA_WARP_SIZE == 0) ? keypoints_num / CUDA_WARP_SIZE : keypoints_num / CUDA_WARP_SIZE + 1;
        dim3 blocks(calc_block_size(previous_frame->h_valid_keypoints_num, threads.x));
        kernel_reproject_prev_points<<<blocks, threads, 0, stream>>>(d_pos_tmp,
                                                                     previous_frame->d_points,
                                                                     previous_frame->h_valid_keypoints_num,
                                                                     T_w2c_prev_curr_,
                                                                     d_rgb_intrin);

        kernel_match_keypoints<<<blocks, threads, 0, stream>>>(d_pos_tmp,
                                                               current_frame->d_pos,
                                                               d_pos_frame,
                                                               previous_frame->d_descriptors,
                                                               current_frame->d_descriptors,
                                                               max_pixel_distance,
                                                               max_hamming_distance,
                                                               previous_frame->h_valid_keypoints_num,
                                                               d_valid_keypoints_num,
                                                               previous_matched_points,
                                                               current_matched_points,
                                                               (double3 *)previous_frame->d_points,
                                                               (double3 *)current_frame->d_points,
                                                               previous_frame->keypoints_count,
                                                               d_keypoints_num_matched);

        // checkCudaErrors(cudaStreamSynchronize(stream)); // to get h_keypoints_num_matched

        checkCudaErrors(cudaMemcpyAsync((void *)h_previous_matched_points,
                                        previous_matched_points,
                                        sizeof(double3) * previous_frame->keypoints_count,
                                        cudaMemcpyDeviceToHost,
                                        stream));

        checkCudaErrors(cudaMemcpyAsync((void *)h_current_matched_points,
                                        current_matched_points,
                                        sizeof(double3) * previous_frame->keypoints_count,
                                        cudaMemcpyDeviceToHost,
                                        stream));

        checkCudaErrors(cudaMemcpyAsync((void *)h_keypoints_num_matched,
                                        d_keypoints_num_matched,
                                        sizeof(int),
                                        cudaMemcpyDeviceToHost,
                                        stream));
        checkCudaErrors(cudaStreamSynchronize(stream)); // to get h_keypoints_num_matched

        current_frame->keypoints_x = (uint16_t *)malloc(sizeof(uint16_t) * (*h_keypoints_num_matched));
        current_frame->keypoints_y = (uint16_t *)malloc(sizeof(uint16_t) * (*h_keypoints_num_matched));

        checkCudaErrors(cudaMemcpyAsync((void *)current_frame->keypoints_x,
                                        d_pos_frame,
                                        sizeof(uint16_t) * (*h_keypoints_num_matched),
                                        cudaMemcpyDeviceToHost,
                                        stream));

        checkCudaErrors(cudaMemcpyAsync((void *)current_frame->keypoints_y,
                                        d_pos_frame + previous_frame->keypoints_count,
                                        sizeof(uint16_t) * (*h_keypoints_num_matched),
                                        cudaMemcpyDeviceToHost,
                                        stream));
        checkCudaErrors(cudaStreamSynchronize(stream)); // to get h_keypoints_num_matched

        std::cout << "h_keypoints_num_matched " << *h_keypoints_num_matched << std::endl;

        checkCudaErrors(cudaFree(T_w2c_prev_curr_));
        checkCudaErrors(cudaFree(d_pos_tmp));
        checkCudaErrors(cudaFree(d_pos_frame));
        checkCudaErrors(cudaFree(previous_matched_points));
        checkCudaErrors(cudaFree(current_matched_points));
        std::cout << "freed memory in post_processing" << std::endl;
    }
}