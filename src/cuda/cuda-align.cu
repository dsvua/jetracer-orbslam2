
// #include "cuda-align.cuh"
// #include "../../../include/librealsense2/rsutil.h"
#include "rscuda_utils.cuh"

// CUDA headers
// #include <cuda_runtime.h>
#include "cuda-align.cuh"


#ifdef _MSC_VER 
// Add library dependencies if using VS
#pragma comment(lib, "cudart_static")
#endif

#define RS2_CUDA_THREADS_PER_BLOCK 32

// using namespace rscuda;

template<int N> struct bytes { unsigned char b[N]; };

int calc_block_size(int pixel_count, int thread_count)
{
    return ((pixel_count % thread_count) == 0) ? (pixel_count / thread_count) : (pixel_count / thread_count + 1);
}

__device__ void kernel_transfer_pixels(int2* mapped_pixels, const rs2_intrinsics* depth_intrin,
    const rs2_intrinsics* other_intrin, const rs2_extrinsics* depth_to_other, float depth_val, int depth_x, int depth_y, int block_index)
{
    float shift = block_index ? 0.5 : -0.5;
    auto depth_size = depth_intrin->width * depth_intrin->height;
    auto mapped_index = block_index * depth_size + (depth_y * depth_intrin->width + depth_x);

    if (mapped_index >= depth_size * 2)
        return;

    // Skip over depth pixels with the value of zero, we have no depth data so we will not write anything into our aligned images
    if (depth_val == 0)
    {
        mapped_pixels[mapped_index] = { -1, -1 };
        return;
    }

    //// Map the top-left corner of the depth pixel onto the other image
    float depth_pixel[2] = { depth_x + shift, depth_y + shift }, depth_point[3], other_point[3], other_pixel[2];
    rscuda::rs2_deproject_pixel_to_point(depth_point, depth_intrin, depth_pixel, depth_val);
    rscuda::rs2_transform_point_to_point(other_point, depth_to_other, depth_point);
    rscuda::rs2_project_point_to_pixel(other_pixel, other_intrin, other_point);
    mapped_pixels[mapped_index].x = static_cast<int>(other_pixel[0] + 0.5f);
    mapped_pixels[mapped_index].y = static_cast<int>(other_pixel[1] + 0.5f);
}

__global__  void kernel_map_depth_to_other(int2* mapped_pixels, const uint16_t* depth_in, const rs2_intrinsics* depth_intrin, const rs2_intrinsics* other_intrin,
    const rs2_extrinsics* depth_to_other, float depth_scale)
{
    int depth_x = blockIdx.x * blockDim.x + threadIdx.x;
    int depth_y = blockIdx.y * blockDim.y + threadIdx.y;

    int depth_pixel_index = depth_y * depth_intrin->width + depth_x;
    if (depth_pixel_index >= depth_intrin->width * depth_intrin->height)
        return;
    float depth_val = depth_in[depth_pixel_index] * depth_scale;
    kernel_transfer_pixels(mapped_pixels, depth_intrin, other_intrin, depth_to_other, depth_val, depth_x, depth_y, blockIdx.z);
}

template<int BPP>
__global__  void kernel_other_to_depth(unsigned char* aligned, const unsigned char* other, const int2* mapped_pixels, const rs2_intrinsics* depth_intrin, const rs2_intrinsics* other_intrin)
{
    int depth_x = blockIdx.x * blockDim.x + threadIdx.x;
    int depth_y = blockIdx.y * blockDim.y + threadIdx.y;

    auto depth_size = depth_intrin->width * depth_intrin->height;
    int depth_pixel_index = depth_y * depth_intrin->width + depth_x;

    if (depth_pixel_index >= depth_intrin->width * depth_intrin->height)
        return;

    int2 p0 = mapped_pixels[depth_pixel_index];
    int2 p1 = mapped_pixels[depth_size + depth_pixel_index];

    if (p0.x < 0 || p0.y < 0 || p1.x >= other_intrin->width || p1.y >= other_intrin->height)
        return;

    // Transfer between the depth pixels and the pixels inside the rectangle on the other image
    auto in_other = (const bytes<BPP> *)(other);
    auto out_other = (bytes<BPP> *)(aligned);
    for (int y = p0.y; y <= p1.y; ++y)
    {
        for (int x = p0.x; x <= p1.x; ++x)
        {
            auto other_pixel_index = y * other_intrin->width + x;
            out_other[depth_pixel_index] = in_other[other_pixel_index];
        }
    }
}

__global__  void kernel_depth_to_other(uint16_t* aligned_out, const uint16_t* depth_in, const int2* mapped_pixels, const rs2_intrinsics* depth_intrin, const rs2_intrinsics* other_intrin)
{
    int depth_x = blockIdx.x * blockDim.x + threadIdx.x;
    int depth_y = blockIdx.y * blockDim.y + threadIdx.y;

    auto depth_size = depth_intrin->width * depth_intrin->height;
    int depth_pixel_index = depth_y * depth_intrin->width + depth_x;

    if (depth_pixel_index >= depth_intrin->width * depth_intrin->height)
        return;

    int2 p0 = mapped_pixels[depth_pixel_index];
    int2 p1 = mapped_pixels[depth_size + depth_pixel_index];

    if (p0.x < 0 || p0.y < 0 || p1.x >= other_intrin->width || p1.y >= other_intrin->height)
        return;

    // Transfer between the depth pixels and the pixels inside the rectangle on the other image
    unsigned int new_val = depth_in[depth_pixel_index];
    unsigned int* arr = (unsigned int*)aligned_out;
    for (int y = p0.y; y <= p1.y; ++y)
    {
        for (int x = p0.x; x <= p1.x; ++x)
        {
            auto other_pixel_index = y * other_intrin->width + x;
            new_val = new_val << 16 | new_val;
            atomicMin(&arr[other_pixel_index / 2], new_val);
        }
    }
}

__global__  void kernel_replace_to_zero(uint16_t* aligned_out, const rs2_intrinsics* other_intrin)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    auto other_pixel_index = y * other_intrin->width + x;
    if (aligned_out[other_pixel_index] == 0xffff)
        aligned_out[other_pixel_index] = 0;
}
