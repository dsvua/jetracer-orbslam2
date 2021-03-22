#include "post_processing.cuh"
#include "../cuda_common.h"

#include <helper_cuda.h>

namespace Jetracer
{
    __global__ void kernel_overlay_keypoints(unsigned char *d_image,
                                             std::size_t pitch,
                                             int height,
                                             float2 *d_pos,
                                             unsigned int *d_aligned_depth,
                                             int keypoints_num)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        float2 pos = d_pos[idx];

        for (int x = pos.x - 1; x < pos.x + 1; x++)
        {
            for (int y = pos.y - 1; y < pos.y + 1; y++)
            {
                if(d_aligned_depth[y * 848 + x] > 0.1f)
                    d_image[y * pitch + x] = 255;
            }
        }
    }

    void overlay_keypoints(unsigned char *d_image,
                           std::size_t pitch,
                           int height,
                           float2 *d_pos,
                           unsigned int *d_aligned_depth,
                           int keypoints_num,
                           cudaStream_t stream)
    {
        dim3 threads(CUDA_WARP_SIZE);
        int tmp_blocks = (keypoints_num % CUDA_WARP_SIZE == 0) ? keypoints_num / CUDA_WARP_SIZE : keypoints_num / CUDA_WARP_SIZE + 1;
        dim3 blocks(tmp_blocks);
        kernel_overlay_keypoints<<<blocks, threads, 0, stream>>>(d_image,
                                                                 pitch,
                                                                 height,
                                                                 d_pos,
                                                                 d_aligned_depth,
                                                                 keypoints_num);
    }
}