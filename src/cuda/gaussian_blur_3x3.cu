#include "orb.cuh"
#include "../cuda_common.h"

#include <vector>
#include <helper_cuda.h>

namespace Jetracer
{
    /* Gaussian 3x3 kernel
    * 1 2 1
    * 2 4 2 * 1/16
    * 1 2 1
    */

    __global__ void gaussian_blur_3x3_kernel(unsigned char *blurred_image,
                                             int blurred_image_pitch,
                                             unsigned char *image,
                                             int image_pitch,
                                             int image_width,
                                             int image_height)
    {
        if (blockIdx.y == 0 || blockIdx.y >= image_height - 2)
            return;

        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = blockIdx.y;
        int blur_sum = 0;

        if (x < image_width)
        {
            int a = image[(y - 1) * image_pitch + x];
            int b = image[y * image_pitch + x];
            int c = image[(y + 1) * image_pitch + x];

            __syncthreads();

            blur_sum += a * 2;
            blur_sum += b * 4;
            blur_sum += c * 2;
            blur_sum += __shfl_up_sync(FULL_MASK, a, 1);
            blur_sum += __shfl_down_sync(FULL_MASK, a, 1);
            blur_sum += __shfl_up_sync(FULL_MASK, b, 1) * 2;
            blur_sum += __shfl_down_sync(FULL_MASK, b, 1) * 2;
            blur_sum += __shfl_up_sync(FULL_MASK, c, 1);
            blur_sum += __shfl_down_sync(FULL_MASK, c, 1);
    
            __syncthreads();
    
            blurred_image[y * blurred_image_pitch + x] = floor(float(blur_sum) / 16.0f + 0.5);

        }

    }

    void gaussian_blur_3x3(unsigned char *blurred_image,
                           int blurred_image_pitch,
                           unsigned char *image,
                           int image_pitch,
                           int image_width,
                           int image_height,
                           cudaStream_t stream)
    {
        //launch kernel
        dim3 threads(CUDA_WARP_SIZE, 1);
        dim3 blocks(calc_block_size(image_width, 30), image_height);
        gaussian_blur_3x3_kernel<<<blocks, threads, 0, stream>>>(blurred_image,
                                                                 blurred_image_pitch,
                                                                 image,
                                                                 image_pitch,
                                                                 image_width,
                                                                 image_height);
        // CUDA_KERNEL_CHECK();
    }
}