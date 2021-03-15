#pragma once

#ifndef JETRACER_RSCUDA_UTILS_H
#define JETRACER_RSCUDA_UTILS_H

// CUDA headers
#include <cuda_runtime.h>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

#ifdef _MSC_VER
// Add library dependencies if using VS
#pragma comment(lib, "cudart_static")
#endif

namespace rscuda
{

    // using namespace librealsense;

    template <typename T>
    std::shared_ptr<T> alloc_dev(int elements)
    {
        T *d_data;
        auto res = cudaMalloc(&d_data, sizeof(T) * elements);
        if (res != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed status: " + res);
        return std::shared_ptr<T>(d_data, [](T *p) { cudaFree(p); });
    }

    template <typename T>
    std::shared_ptr<T> make_device_copy(T obj)
    {
        T *d_data;
        auto res = cudaMalloc(&d_data, sizeof(T));
        if (res != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed status: " + res);
        cudaMemcpy(d_data, &obj, sizeof(T), cudaMemcpyHostToDevice);
        return std::shared_ptr<T>(d_data, [](T *data) { cudaFree(data); });
    }

    /* Given a point in 3D space, compute the corresponding pixel coordinates in an image with no distortion or forward distortion coefficients produced by the same camera */
    __device__ static void rs2_project_point_to_pixel(float pixel[2], const struct rs2_intrinsics *intrin, const float point[3]);

    /* Given pixel coordinates and depth in an image with no distortion or inverse distortion coefficients, compute the corresponding point in 3D space relative to the same camera */
    __device__ static void rs2_deproject_pixel_to_point(float point[3], const struct rs2_intrinsics *intrin, const float pixel[2], float depth);

    /* Transform 3D coordinates relative to one sensor to 3D coordinates relative to another viewpoint */
    __device__ static void rs2_transform_point_to_point(float to_point[3], const struct rs2_extrinsics *extrin, const float from_point[3]);
} // namespace rscuda

#endif // JETRACER_RSCUDA_UTILS_H
