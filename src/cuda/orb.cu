#include "orb.cuh"
#include "../cuda_common.h"

#include <vector>
#include <helper_cuda.h>

namespace Jetracer
{

    __constant__ unsigned char c_pattern[sizeof(int2) * 512];

#define GET_VALUE(idx)                                                                      \
    image[(loc.y + __float2int_rn(pattern[idx].x * b + pattern[idx].y * a)) * image_pitch + \
          loc.x + __float2int_rn(pattern[idx].x * a - pattern[idx].y * b)]

    // __global__ void calcOrb_kernel(const PtrStepb image, float2 *d_keypoints_pos, const int npoints, PtrStepb descriptors)
    __global__ void calc_orb_kernel(float *d_keypoints_angle,
                                    float2 *d_keypoints_pos,
                                    unsigned char *d_descriptors,
                                    unsigned char *image,
                                    int image_pitch,
                                    int image_width,
                                    int image_height,
                                    int keypoints_num)
    {
        int id = blockIdx.x;
        int tid = threadIdx.x;
        if (id >= keypoints_num)
            return;

        const float2 kpt = d_keypoints_pos[id];
        short2 loc = make_short2(short(kpt.x), short(kpt.y));
        unsigned char *desc = d_descriptors;
        if (loc.x < 17 || loc.x > image_width - 17 || loc.y < 17 || loc.y > image_height - 17)
        {
            desc[id * 32 + tid] = 0;
            return;
        }

        const int2 *pattern = ((int2 *)c_pattern) + 16 * tid;

        const float factorPI = (float)(CUDART_PI_F / 180.f);
        float angle = d_keypoints_angle[id] * factorPI;

        float a = (float)cosf(angle);
        float b = (float)sinf(angle);

        int t0, t1, val;
        t0 = GET_VALUE(0);
        t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2);
        t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4);
        t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6);
        t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8);
        t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10);
        t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12);
        t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14);
        t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[id * 32 + tid] = (unsigned char)val;
    }

    __global__ void compute_fast_angle_kernel(float *d_keypoints_angle,
                                              float2 *d_keypoints_pos,
                                              unsigned char *image,
                                              int image_pitch,
                                              int image_width,
                                              int image_height,
                                              int keypoints_num)
    {
        int idx = blockIdx.x;
        int k_x = floor(d_keypoints_pos[idx].x + 0.5);
        int k_y = floor(d_keypoints_pos[idx].y + 0.5);
        // Hardcoding for patch 31x31, so, radius is 15
        int r2 = 15 * 15;
        float m10 = 0;
        float m01 = 0;

        // Hardcoding for patch 31x31, so, radius is 15
        if (threadIdx.x < 31)
        {
            int mult_dx = threadIdx.x - 15;
            int tdx = threadIdx.x + k_x - 15;
            if (tdx > 0 && tdx < image_width)
            {
                m10 = mult_dx * image[k_y * image_pitch + tdx];
            }
        }

        for (int dy = 1; dy < 16; dy++)
        {
            int dx = floor(sqrtf(r2 - float(dy * dy)) + 0.5);
            if (threadIdx.x > 14 - dx && threadIdx.x < 16 + dx)
            {
                int mult_dx = threadIdx.x - 15;
                int tdx = k_x + threadIdx.x - 15;

                if (k_y - dy > 0 && tdx > 0 && tdx < image_width)
                {
                    float i = image[(k_y - dy) * image_pitch + tdx];
                    m01 -= dy * i;
                    m10 += mult_dx * i;
                }

                if (k_y + dy < image_height && tdx > 0 && tdx < image_width)
                {
                    float i = image[(k_y + dy) * image_pitch + tdx];
                    m01 += dy * i;
                    m10 += mult_dx * i;
                }
            }
        }

        __syncthreads();

        for (int offset = 16; offset > 0; offset /= 2)
        {
            m01 += __shfl_down_sync(FULL_MASK, m01, offset);
            m10 += __shfl_down_sync(FULL_MASK, m10, offset);
        }

        __syncthreads();

        if (threadIdx.x == 0)
        {
            d_keypoints_angle[idx] = atan2f(m01, m10);
        }
    }

    // compress 32 x chars to 1 x uint32_t to be able to use bitwise operations
    __global__ void compress_descriptors_kernel(unsigned char *d_descriptors_tmp,
                                                uint32_t *d_descriptors,
                                                int keypoints_num)
    {
        uint32_t descriptor = 0;
        uint32_t d_one = 1;
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < keypoints_num)
        {
            for (int i = 0; i < 32; i++)
            {
                if (d_descriptors_tmp[idx * 32 + i] == 1)
                {
                    descriptor = descriptor | (d_one << i);
                }
            }
        }

        __syncthreads();

        if (idx < keypoints_num)
        {
            d_descriptors[idx] = descriptor;
        }
    }

    void compute_fast_angle(float *d_keypoints_angle,
                            float2 *d_keypoints_pos,
                            unsigned char *image,
                            int image_pitch,
                            int image_width,
                            int image_height,
                            int keypoints_num,
                            cudaStream_t stream)
    {
        //sdfs
        compute_fast_angle_kernel<<<keypoints_num, 32, 0, stream>>>(d_keypoints_angle,
                                                                    d_keypoints_pos,
                                                                    image,
                                                                    image_pitch,
                                                                    image_width,
                                                                    image_height,
                                                                    keypoints_num);
        // CUDA_KERNEL_CHECK();
    }

    void calc_orb(float *d_keypoints_angle,
                  float2 *d_keypoints_pos,
                  unsigned char *d_descriptors_tmp,
                  uint32_t *d_descriptors,
                  unsigned char *image,
                  int image_pitch,
                  int image_width,
                  int image_height,
                  int keypoints_num,
                  cudaStream_t stream)
    {
        calc_orb_kernel<<<keypoints_num, CUDA_WARP_SIZE, 0, stream>>>(d_keypoints_angle,
                                                                      d_keypoints_pos,
                                                                      d_descriptors_tmp,
                                                                      image,
                                                                      image_pitch,
                                                                      image_width,
                                                                      image_height,
                                                                      keypoints_num);
        // CUDA_KERNEL_CHECK();
        dim3 threads(CUDA_WARP_SIZE);
        dim3 blocks(calc_block_size(keypoints_num, threads.x));
        compress_descriptors_kernel<<<blocks, threads, 0, stream>>>(d_descriptors_tmp,
                                                                    d_descriptors,
                                                                    keypoints_num);
    }

    void loadPattern()
    {
        const int npoints = 512;
        std::vector<int2> pattern;
        const int2 *pattern0 = (const int2 *)bit_pattern_31_;
        std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));
        checkCudaErrors(cudaMemcpyToSymbol(c_pattern, pattern.data(), sizeof(int2) * 512));
    }
} // namespace Jetracer
