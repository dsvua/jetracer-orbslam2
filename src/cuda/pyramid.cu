#include "pyramid.cuh"
#include "../cuda_common.h"

namespace Jetracer
{
    template <typename T, const int N>
    __global__ void image_halfsample_gpu_kernel(const uchar2 *__restrict__ d_image_in,
                                                const unsigned int pitch_src_px,
                                                T *__restrict__ d_image_out,
                                                const unsigned int width_px,
                                                const unsigned int height_px,
                                                const unsigned int pitch_dst_px)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if ((x < width_px) && (y < height_px))
        {
            const int dst = y * pitch_dst_px / N + x; //every thread writes N bytes. the next row starts at pitch_dst_px/N
            int src_top = y * pitch_src_px + x * N;   //every thread reads in Nx2 bytes
            int src_bottom = y * pitch_src_px + x * N + (pitch_src_px / 2);
#pragma unroll N
            for (int i = 0; i < N; ++i)
            {
                const uchar2 t2 = d_image_in[src_top++];
                const uchar2 b2 = d_image_in[src_bottom++];
                *(((unsigned char *)(d_image_out + dst)) + i) = (unsigned char)(((unsigned int)t2.x + (unsigned int)t2.y + (unsigned int)b2.x + (unsigned int)b2.y) >> 2);
            }
        }
    }

    void pyramid_create_levels(std::vector<pyramid_t> pyramid,
                               cudaStream_t stream)
    {
        for (int i = 1; i < pyramid.size(); i++)
        {
            // Use the most efficient vectorized version
            for (unsigned int v = 4; v > 0; v = v / 2)
            {
                // std::cout << "<---- if (pyramid[i].image_width % v == 0)" << std::endl;
                if (pyramid[i].image_width % v == 0)
                {
                    const unsigned int img_dst_width_n = pyramid[i].image_width / v;
                    const unsigned int thread_num_x = min(64, ((img_dst_width_n + 32 - 1) / 32) * 32);
                    dim3 threads(thread_num_x, 2);
                    dim3 blocks((img_dst_width_n + thread_num_x - 1) / threads.x, (pyramid[i].image_height + 2 - 1) / threads.y);
                    switch (v)
                    {
                    case 1:
                        // std::cout << "<---- image_halfsample_gpu_kernel<uchar1, 1>" << std::endl;
                        image_halfsample_gpu_kernel<uchar1, 1><<<blocks, threads, 0, stream>>>(
                            (uchar2 *)pyramid[i - 1].image,
                            (unsigned int)pyramid[i - 1].image_pitch,
                            (uchar1 *)pyramid[i].image,
                            (unsigned int)pyramid[i].image_width,
                            (unsigned int)pyramid[i].image_height,
                            (unsigned int)pyramid[i].image_pitch);
                        break;
                    case 2:
                        // std::cout << "<---- image_halfsample_gpu_kernel<uchar2, 2>" << std::endl;
                        image_halfsample_gpu_kernel<uchar2, 2><<<blocks, threads, 0, stream>>>(
                            (uchar2 *)pyramid[i - 1].image,
                            (unsigned int)pyramid[i - 1].image_pitch,
                            (uchar2 *)pyramid[i].image,
                            (unsigned int)(img_dst_width_n),
                            (unsigned int)(pyramid[i].image_height),
                            (unsigned int)(pyramid[i].image_pitch));
                        break;
                    case 4:
                        // std::cout << "<---- image_halfsample_gpu_kernel<uchar4, 4>" << std::endl;
                        image_halfsample_gpu_kernel<uchar4, 4><<<blocks, threads, 0, stream>>>(
                            (uchar2 *)pyramid[i - 1].image,
                            (unsigned int)pyramid[i - 1].image_pitch,
                            (uchar4 *)pyramid[i].image,
                            (unsigned int)(img_dst_width_n),
                            (unsigned int)(pyramid[i].image_height),
                            (unsigned int)(pyramid[i].image_pitch));
                        break;
                    }
                    break;
                }
            }
        }
        // checkCudaErrors(cudaStreamSynchronize(stream));
    }
} // namespace Jetracer
