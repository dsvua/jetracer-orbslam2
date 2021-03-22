#ifndef JETRACER_POST_PROCESSING_H
#define JETRACER_POST_PROCESSING_H

#include <iostream>
#include <string_view>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace Jetracer
{
    void overlay_keypoints(unsigned char *d_image,
                           std::size_t pitch,
                           int height,
                           float2 *d_pos,
                           unsigned int *d_aligned_depth,
                           int keypoints_num,
                           cudaStream_t stream);
} // namespace Jetracer

#endif // JETRACER_POST_PROCESSING_H