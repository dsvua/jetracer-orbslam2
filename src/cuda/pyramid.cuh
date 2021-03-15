#ifndef JETRACER_CUDA_PYRAMID_KEYPOINTS_H
#define JETRACER_CUDA_PYRAMID_KEYPOINTS_H

#include <vector>
#include <cuda_runtime.h>

namespace Jetracer
{
    typedef struct
    {
        std::size_t image_width;
        std::size_t image_height;
        std::size_t image_pitch;
        unsigned char *image;

        std::size_t response_pitch;
        float *response;
    } pyramid_t;

    void pyramid_create_levels(std::vector<pyramid_t> pyramid,
                               cudaStream_t stream);

} // namespace Jetracer

#endif // JETRACER_CUDA_PYRAMID_KEYPOINTS_H
