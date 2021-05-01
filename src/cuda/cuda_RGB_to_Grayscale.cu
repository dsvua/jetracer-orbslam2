#include "cuda_RGB_to_Grayscale.cuh"
#include "cuda-align.cuh"
#include "../cuda_common.h"
#include <iostream>

#define RS2_CUDA_THREADS_PER_BLOCK 32

namespace Jetracer
{
    __global__ void kernel_rgb_to_grayscale(unsigned char *dst, unsigned char *src, int cols, int rows, int dst_pitch, int src_pitch)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < cols && y < rows)
        {
            float R, G, B;
            R = float(src[y * src_pitch + x * 3 + 0]);
            G = float(src[y * src_pitch + x * 3 + 1]);
            B = float(src[y * src_pitch + x * 3 + 2]);
            dst[y * dst_pitch + x] = floor((B*0.07 + G*0.72 + R* 0.21) + 0.5);
        }
    }

    void rgb_to_grayscale(unsigned char *dst, unsigned char *src, int cols, int rows, int dst_pitch, int src_pitch, cudaStream_t stream)
    {
        dim3 threads(CUDA_WARP_SIZE, CUDA_WARP_SIZE);
        dim3 blocks(calc_block_size(cols, threads.x), calc_block_size(rows, threads.y));
        // std::cout << "---Mark---kernel_rgb_to_grayscale cols: " << cols << " rows " << rows << std::endl;

        kernel_rgb_to_grayscale<<<blocks, threads, 0, stream>>>(dst, src, cols, rows, dst_pitch, src_pitch);
        // CUDA_KERNEL_CHECK();
    }
}
