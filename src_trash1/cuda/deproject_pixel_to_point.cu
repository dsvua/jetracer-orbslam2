
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include "cuda_utils.h"
#include "deproject_pixel_to_point.h"
#include <librealsense2/rs.hpp>

namespace Jetracer {
    __global__ void gpu_deproject_pixel_to_point(float3* d_point_cloud, uint16_t* d_depth_image,
            const struct rs2_intrinsics * d_intrinsics, d_context_t* d_ctx)
    {
        int ix = blockIdx.x * blockDim.x + threadIdx.x + d_ctx->left_gap;
        int iy = blockIdx.y * blockDim.y + threadIdx.y;

        float depth = static_cast< float >d_depth_image[iy*width+ix];

        assert(d_intrinsics->model != RS2_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image

        float x = (ix - d_intrinsics->ppx) / d_intrinsics->fx;
        float y = (iy - d_intrinsics->ppy) / d_intrinsics->fy;
        if(d_intrinsics->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
        {
            float r2  = x*x + y*y;
            float f = 1 + d_intrinsics->coeffs[0]*r2 + d_intrinsics->coeffs[1]*r2*r2 + d_intrinsics->coeffs[4]*r2*r2*r2;
            float ux = x*f + 2*d_intrinsics->coeffs[2]*x*y + d_intrinsics->coeffs[3]*(r2 + 2*x*x);
            float uy = y*f + 2*d_intrinsics->coeffs[3]*x*y + d_intrinsics->coeffs[2]*(r2 + 2*y*y);
            x = ux;
            y = uy;
        }
        if (d_intrinsics->model == RS2_DISTORTION_KANNALA_BRANDT4)
        {
            float rd = sqrtf(x*x + y*y);
            if (rd < FLT_EPSILON)
            {
                rd = FLT_EPSILON;
            }

            float theta = rd;
            float theta2 = rd*rd;
            for (int i = 0; i < 4; i++)
            {
                float f = theta*(1 + theta2*(d_intrinsics->coeffs[0] + theta2*(d_intrinsics->coeffs[1] + theta2*(d_intrinsics->coeffs[2] + theta2*d_intrinsics->coeffs[3])))) - rd;
                if (abs(f) < FLT_EPSILON)
                {
                    break;
                }
                float df = 1 + theta2*(3 * d_intrinsics->coeffs[0] + theta2*(5 * d_intrinsics->coeffs[1] + theta2*(7 * d_intrinsics->coeffs[2] + 9 * theta2*d_intrinsics->coeffs[3])));
                theta -= f / df;
                theta2 = theta*theta;
            }
            float r = tan(theta);
            x *= r / rd;
            y *= r / rd;
        }
        if (d_intrinsics->model == RS2_DISTORTION_FTHETA)
        {
            float rd = sqrtf(x*x + y*y);
            if (rd < FLT_EPSILON)
            {
                rd = FLT_EPSILON;
            }
            float r = (float)(tan(d_intrinsics->coeffs[0] * rd) / atan(2 * tan(d_intrinsics->coeffs[0] / 2.0f)));
            x *= r / rd;
            y *= r / rd;
        }

        d_point_cloud[iy*width+ix-d_ctx->left_gap].x = depth * x;
        d_point_cloud[iy*width+ix-d_ctx->left_gap].y = depth * y;
        d_point_cloud[iy*width+ix-d_ctx->left_gap].z = depth;
    }


    void deproject_pixel_to_point(context_t* ctx, rs2::depth_frame depth_frame){
         // Set flag to enable zero copy access
        cudaError_t err = cudaSuccess;

        d_context_t h_ctx;
        h_ctx.width = ctx->cam_w;
        h_ctx.height = ctx->cam_h;
        h_ctx.left_gap = ctx->left_gap;
        h_ctx.bottom_gap = ctx->bottom_gap;
        h_ctx.min_obstacle_height = ctx->min_obstacle_height;
        h_ctx.max_obstacle_height = ctx->max_obstacle_height;
        // h_ctx.jetson_camera_intrinsics = ctx->jetson_camera_intrinsics;
        // h_ctx. = ctx->;

        d_context_t* d_ctx;
        err = cudaMalloc(&d_ctx, sizeof(d_context_t));
        checkCudaError(err, "cudaMalloc");
        err = cudaMemcpy(d_ctx, &h_ctx, sizeof(d_context_t), cudaMemcpyHostToDevice);
        checkCudaError(err, "cudaMemcpy");

        rs2::rs2_intrinsics* d_intrinsics;
        err = cudaMalloc(&d_intrinsics, sizeof(rs2::rs2_intrinsics));
        checkCudaError(err, "cudaMalloc");
        err = cudaMemcpy(d_intrinsics, ctx->jetson_camera_intrinsics, sizeof(rs2::rs2_intrinsics), cudaMemcpyHostToDevice);
        checkCudaError(err, "cudaMemcpy");

        size_t mallocSize = h_ctx.width*h_ctx.height*sizeof(uint16_t);     
        uint16_t* d_depth_image;
        err = cudaMalloc(&d_depth_image, mallocSize);
        checkCudaError(err, "cudaMalloc");
        err = cudaMemcpy(d_depth_image, (void*)depth_frame.get_data(), mallocSize, cudaMemcpyHostToDevice);
        checkCudaError(err, "cudaMemcpy");
        
        float3* d_point_cloud;
        size_t point_cloud_size = (h_ctx.width-left_gap)*(h_ctx.height-bottom_gap)*sizeof(float3);
        err = cudaMalloc(&d_point_cloud, point_cloud_size);
        checkCudaError(err, "cudaMalloc");

        dim3 block(8, 8);
        dim3 grid((h_ctx.width-left_gap)/block.x+1,(h_ctx.height-bottom_gap)/block.y+1);

        printf("Calling gpu_deproject_pixel_to_point kernel...\n");
        gpu_deproject_pixel_to_point<<<grid, block>>>(d_point_cloud, d_depth_image, d_intrinsics, d_ctx);
        err = cudaGetLastError();
        checkCudaError(err, "gpu_deproject_pixel_to_point kernel");

        float3* h_point_cloud;
        h_point_cloud = (float3*)malloc(point_cloud_size);
        err = cudaMemcpy(h_point_cloud, d_point_cloud, point_cloud_size, cudaMemcpyDeviceToHost);
        checkCudaError(err, "cudaMemcpy");

        // err = cudaDeviceSynchronize();
        // checkCudaError(err, "cudaDeviceSynchronize");

        // printf("Dumping frame to image %d \n", frameNumber);

        // char filename_bin[256];
        // sprintf(filename_bin, "/media/94a96ba3-cd42-4f36-b3e4-d264669645a6/images/output%03u.bin", frameNumber);
        // rawImageSaver(width, height*2, image, filename_bin); // height*2 as we have two images

        // printf("Freeing d_image memory\n");
        // cudaFree(d_image);
        // printf("Freeing image memory\n");
        // delete image;
       
    }
}  // namespace Jetracer
