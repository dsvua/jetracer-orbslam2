#ifndef JETRACER_CUDA_FAST_KEYPOINTS_H
#define JETRACER_CUDA_FAST_KEYPOINTS_H

#include <vector>
#include <cuda_runtime.h>

#include "pyramid.cuh"

// FAST detector parameters
// #define FAST_EPSILON (12.0f)
// #define FAST_MIN_ARC_LENGTH 12
// Remark: the Rosten CPU version only works with
//         SUM_OF_ABS_DIFF_ON_ARC and MAX_THRESHOLD
// #define FAST_SCORE SUM_OF_ABS_DIFF_ON_ARC

namespace Jetracer
{
    enum fast_score
    {
        SUM_OF_ABS_DIFF_ALL = 0, // OpenCV: https://docs.opencv.org/master/df/d0c/tutorial_py_fast.html
        SUM_OF_ABS_DIFF_ON_ARC,  // Rosten 2006
        MAX_THRESHOLD            // Rosten 2008
    };

    void fast_gpu_calculate_lut(unsigned char *d_corner_lut,
                                const int &min_arc_length);

    void fast_gpu_calc_corner_response(const int image_width,
                                       const int image_height,
                                       const int image_pitch,
                                       const unsigned char *d_image,
                                       const int horizontal_border,
                                       const int vertical_border,
                                       const unsigned char *d_corner_lut,
                                       const float threshold,
                                       const int min_arc_length,
                                       const fast_score score,
                                       const int response_pitch_elements,
                                       float *d_response,
                                       cudaStream_t stream);

    void detect(std::vector<pyramid_t> pyramid,
                const unsigned char *d_corner_lut,
                const float threshold,
                float2 *d_pos,
                float *d_score,
                int *d_level,
                cudaStream_t stream);
} // namespace Jetracer

#endif // JETRACER_CUDA_FAST_KEYPOINTS_H
