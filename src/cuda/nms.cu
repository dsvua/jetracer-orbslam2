#include "nms.cuh"
#include "../cuda_common.h"
#include "../SlamGpuPipeline/defines.h"

namespace Jetracer
{

    __inline__ __device__ void nms_offset_pos(const int &i, float &x_offset, float &y_offset)
    {
        /*
        * 3x3 (n=1):
        * 7 0 1
        * 6 x 2
        * 5 4 3
        */
        if (i == 0)
        {
            x_offset = 0.0f;
            y_offset = -1.0f;
        }
        if (i == 1)
        {
            x_offset = 1.0f;
            y_offset = -1.0f;
        }
        if (i == 2)
        {
            x_offset = 1.0f;
            y_offset = 0.0f;
        }
        if (i == 3)
        {
            x_offset = 1.0f;
            y_offset = 1.0f;
        }
        if (i == 4)
        {
            x_offset = 0.0f;
            y_offset = 1.0f;
        }
        if (i == 5)
        {
            x_offset = -1.0f;
            y_offset = 1.0f;
        }
        if (i == 6)
        {
            x_offset = -1.0f;
            y_offset = 0.0f;
        }
        if (i == 7)
        {
            x_offset = -1.0f;
            y_offset = -1.0f;
        }
    }

    __inline__ __device__ int nms_offset(const int &i, const int &pitch)
    {
        int offs = 0;
        /*
        * 3x3 (n=1):
        * 7 0 1
        * 6 x 2
        * 5 4 3
        */
        if (i == 0)
            offs = -pitch;
        if (i == 1)
            offs = -pitch + 1;
        if (i == 2)
            offs = +1;
        if (i == 3)
            offs = pitch + 1;
        if (i == 4)
            offs = pitch;
        if (i == 5)
            offs = pitch - 1;
        if (i == 6)
            offs = -1;
        if (i == 7)
            offs = -pitch - 1;
        return offs;
    }

    template <bool strictly_greater>
    __global__ void detector_base_gpu_grid_nms_kernel(const int level,
                                                      const int min_level,
                                                      const int image_width,
                                                      const int image_height,
                                                      const int horizontal_border,
                                                      const int vertical_border,
                                                      const int cell_size_width,
                                                      const int cell_size_height,
                                                      const int response_pitch_elements,
                                                      const float *__restrict__ d_response,
                                                      float2 *__restrict__ d_pos,
                                                      float *__restrict__ d_score,
                                                      int *__restrict__ d_level)
    {
        // Various identifiers
        const int x = cell_size_width * blockIdx.x + threadIdx.x;
        const int y = cell_size_height * blockIdx.y + threadIdx.y;
        const int cell_id = gridDim.x * blockIdx.y + blockIdx.x;
        const int thread_id = threadIdx.x + blockDim.x * threadIdx.y;
        const int lane_id = thread_id & 0x1F;
        const int warp_id = thread_id >> 5;
        const int warp_cnt = (blockDim.x * blockDim.y + 31) >> 5;
        // Selected maximum response
        float max_x = static_cast<float>(x);
        float max_y = 0.f;
        float max_resp = 0.0f;

        if (x < image_width && y < image_height)
        {
            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                // the very first thread in the threadblock, initializes the cell score
                if (level == min_level)
                {
                    d_score[cell_id] = 0.0f;
                }
            }

            /*
            * Note to future self:
            * basically, we perform NMS on every line in the cell just like a regular NMS, BUT
            * we go in a spiral and check if any of the neigbouring values is higher than our current one.
            * If it is higher, than we set our current value to zero.
            * We DO NOT write to the response buffer, we keep everything in registers. Also,
            * we do not use if-else for checking values, we use my signbit trick. This latter is amazing, because
            * it completely avoids warp divergence.
            * Then once all threads in a warp figured out whether the value they are looking at was supressed
            * or not, they reduce with warp-level intrinsics to lane 0.
            */

            // Maximum value
            // Location: x,y -> x is always threadIdx.x
            int max_y_tmp = 0;

            // border remains the same irrespective of pyramid level
            int image_width_m_border = image_width - horizontal_border;
            int image_height_m_border = image_height - vertical_border;
            if (x >= horizontal_border && x < image_width_m_border)
            {
                // we want as few idle threads as possible, hence we shift them according to y
                // note: we shift down all lines within the block
                int cell_top_to_border = vertical_border - (cell_size_height * blockIdx.y);
                int y_offset = cell_top_to_border > 0 ? cell_top_to_border : 0;
                int gy = y + y_offset;
                int box_line = threadIdx.y + y_offset;
                const float *d_response_ptr = d_response + gy * response_pitch_elements + x;
                int response_offset = blockDim.y * response_pitch_elements;
                for (; (box_line < cell_size_height) && (gy < image_height_m_border); box_line += blockDim.y, d_response_ptr += response_offset, gy += blockDim.y)
                {
                    // acquire the center response value
                    float center_value = d_response_ptr[0];

                    // Perform spiral NMS
#pragma unroll
                    for (int i = 0; i < (DETECTOR_BASE_NMS_SIZE * DETECTOR_BASE_NMS_SIZE - 1); ++i)
                    {
                        int j = nms_offset(i, response_pitch_elements);

                        // Perform non-maximum suppression
                        if (strictly_greater)
                        {
                            center_value *= -0.5f * (-1.0f + copysignf(1.0f, d_response_ptr[j] - center_value));
                        }
                        else
                        {
                            center_value *= 0.5f * (1.0f + copysignf(1.0f, center_value - d_response_ptr[j]));
                        }
                        /*
                        * Note to future self:
                        * Interestingly on Maxwell (960M), checking for equivalence with 0.0f, results
                        * in better runtimes.
                        * However, on Pascal (Tegra X2) this increases the runtime, hence we opted for
                        * not checking it for now.
                        */
                        // #if 0
                        if (center_value == 0.0f)
                            break; // we should check it on the Jetson
                                   // #endif /* 0 */
                    }

                    // NMS is over, is this value greater than the previous one?
                    if (center_value > max_resp)
                    {
                        max_resp = center_value;
                        max_y_tmp = gy;
                    }
                }
            }
            // Perform conversion
            max_y = static_cast<float>(max_y_tmp);
        }

// Reduce the maximum location to thread 0 within each warp
#pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            float max_resp_new = __shfl_down_sync(FULL_MASK, max_resp, offset);
            float max_x_new = __shfl_down_sync(FULL_MASK, max_x, offset);
            float max_y_new = __shfl_down_sync(FULL_MASK, max_y, offset);
            if (max_resp_new > max_resp)
            {
                max_resp = max_resp_new;
                max_x = max_x_new;
                max_y = max_y_new;
            }
        }

        // now each warp's lane 0 has the maximum value of its cell
        // reduce in shared memory
        // each warp's lane 0 writes to shm
        // resp, x, y, (level - not used)
        extern __shared__ float s[];
        float *s_data = s + (warp_id << 2);
        float scale = static_cast<float>(1 << level);

        if (lane_id == 0)
        {
            s_data[0] = max_resp;
            s_data[1] = max_x;
            s_data[2] = max_y;
        }
        __syncthreads();
        // threadId x & y 0 reduces the warp results
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            s_data = s + 4; // skip self results
            for (int i = 1; i < warp_cnt; i++, s_data += 4)
            {
                float max_resp_s = s_data[0];
                float max_x_s = s_data[1];
                float max_y_s = s_data[2];
                if (max_resp_s > max_resp)
                {
                    max_resp = max_resp_s;
                    max_x = max_x_s;
                    max_y = max_y_s;
                }
            }

            if (d_score[cell_id] < max_resp)
            {
                d_score[cell_id] = max_resp;
                d_pos[cell_id].x = max_x * scale;
                d_pos[cell_id].y = max_y * scale;
                d_level[cell_id] = level;
            }
        }
    }

    void grid_nms(std::vector<pyramid_t> pyramid,
                  float2 *d_pos,
                  float *d_score,
                  int *d_level,
                  cudaStream_t stream)
    {
        for (std::size_t level = 0; level < pyramid.size(); ++level)
        {
            const int cell_size_width_level = (CUDA_WARP_SIZE >> level);
            const int cell_size_height_level = (CUDA_WARP_SIZE >> level);
            int target_threads_per_block = 128;
            const int horizontal_cell_num = (pyramid[level].image_width % 32 == 0) ? pyramid[level].image_width / 32 : pyramid[level].image_width / 32 + 1;
            const int vertical_cell_num = (pyramid[level].image_height % 32 == 0) ? pyramid[level].image_height / 32 : pyramid[level].image_height / 32 + 1;

            dim3 threads(cell_size_width_level,
                         max(1, min(target_threads_per_block / cell_size_width_level, cell_size_height_level)));
            dim3 blocks(horizontal_cell_num, vertical_cell_num);

            // shared memory allocation
            int launched_warp_count = (threads.x * threads.y * threads.z + 32 - 1) / 32;
            std::size_t shm_mem_size = launched_warp_count * 4 * sizeof(float);

            detector_base_gpu_grid_nms_kernel<true><<<blocks, threads, shm_mem_size, stream>>>(
                level,
                0,
                pyramid[level].image_width,
                pyramid[level].image_height,
                3,
                3,
                cell_size_width_level,
                cell_size_height_level,
                pyramid[level].response_pitch / sizeof(float),
                pyramid[level].response,
                d_pos,
                d_score,
                d_level);
            // checkCudaErrors(cudaStreamSynchronize(stream));
        }
    }
} // namespace Jetracer