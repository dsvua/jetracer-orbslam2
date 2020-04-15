
#include <cuda.h>

#ifndef JETRACER_CUDA_UTILS_H
#define JETRACER_CUDA_UTILS_H

namespace Jetracer {

    void checkCudaError(cudaError_t err, const char * message){
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to %s (error code %03u %s)!\n",
                             message, err, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }    
    }
}

#endif //JETRACER_CUDA_UTILS_H