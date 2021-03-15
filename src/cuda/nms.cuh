#ifndef JETRACER_CUDA_NMS_KEYPOINTS_H
#define JETRACER_CUDA_NMS_KEYPOINTS_H

#include <vector>
#include <cuda_runtime.h>

#include "pyramid.cuh"

namespace Jetracer
{
    void grid_nms(std::vector<pyramid_t> pyramid,
                  float2 *d_pos,
                  float *d_score,
                  int *d_level,
                  cudaStream_t stream);

} // namespace Jetracer

#endif // JETRACER_CUDA_NMS_KEYPOINTS_H
