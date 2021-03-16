#ifndef JETRACER_RGB_TO_GRAYSCALE_H
#define JETRACER_RGB_TO_GRAYSCALE_H

#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace Jetracer
{
    void rgb_to_grayscale(unsigned char *dst,
                          unsigned char *src,
                          int cols,
                          int rows,
                          int dst_pitch,
                          int src_pitch,
                          cudaStream_t stream);
}

#endif // JETRACER_RGB_TO_GRAYSCALE_H
