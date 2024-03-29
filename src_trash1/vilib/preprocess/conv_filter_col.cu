/*
 * Column-wise convolution filtering
 * conv_filter_col.h
 *
 * Copyright (c) 2019-2020 Balazs Nagy,
 * Robotics and Perception Group, University of Zurich
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "../cuda_common.h"
#include "conv_filter_col.h"

namespace vilib {

#define BLOCKDIM_X               32
#define BLOCKDIM_Y               4
#define RESULT_STEPS             8
#define HALO_STEPS               1

#define INSTANTIATE_1D_COL(I, O)                                                                 \
  template __host__ void conv_filter_col_gpu<I, O>(const I * d_image_in,                         \
                                                   const int input_pitch,                        \
                                                   O * d_image_out,                              \
                                                   const int output_pitch,                       \
                                                   const int width_px,                           \
                                                   const int height_px,                          \
                                                   const conv_filter_type_t filter_type,         \
                                                   const conv_filter_border_type_t border_type,  \
                                                   const bool skip_first_and_last_col,           \
                                                   const float scale,                            \
                                                   cudaStream_t stream)

template<typename I, typename O, int RADIUS, conv_filter_border_type BORDER>
__global__ void conv_filter_col_gpu_shm_kernel(O * __restrict__ output,
                                               const int output_pitch,
                                               const I * __restrict__ input,
                                               const int input_pitch,
                                               const int output_width,
                                               const int input_height,
                                               const filter1x3_t filter,
                                               const float scale) {
  __shared__ float s_Data[BLOCKDIM_X][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCKDIM_Y + 1];

  // Offset to the upper halo edge
  const int baseX = blockIdx.x * BLOCKDIM_X + threadIdx.x;
  const int baseY = (blockIdx.y * RESULT_STEPS - HALO_STEPS) * BLOCKDIM_Y + threadIdx.y;

  if(baseX >= output_width) return;

  input += baseX;
  output += baseY * output_pitch + baseX;

  // Main data
  #pragma unroll
  for (int i = HALO_STEPS, i_y_offset = (i*BLOCKDIM_Y + baseY) * input_pitch; i < HALO_STEPS + RESULT_STEPS; i++, i_y_offset+= BLOCKDIM_Y * input_pitch) {
    s_Data[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y] = input[i_y_offset];
  }

  // Upper halo
  #pragma unroll
  for (int i = 0, i_y_offset = baseY * input_pitch; i < HALO_STEPS; i++, i_y_offset += BLOCKDIM_Y * input_pitch) {
    const int i_y = baseY + i * BLOCKDIM_Y;
    switch(BORDER) {
      case conv_filter_border_type::BORDER_SKIP:
        // fall-through
      case conv_filter_border_type::BORDER_ZERO:          // 000000|abcdefgh|0000000
        s_Data[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y] = (i_y >= 0) ? input[i_y_offset] : 0;
        break;
      case conv_filter_border_type::BORDER_REPLICATE:     // aaaaaa|abcdefgh|hhhhhhh
        s_Data[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y] = (i_y >= 0) ? input[i_y_offset] : input[0];
        break;
      case conv_filter_border_type::BORDER_REFLECT:       // fedcba|abcdefgh|hgfedcb
        s_Data[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y] = (i_y >= 0) ? input[i_y_offset] : input[(-i_y-1) * input_pitch];
        break;
      case conv_filter_border_type::BORDER_WRAP:          // cdefgh|abcdefgh|abcdefg
        s_Data[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y] = (i_y >= 0) ? input[i_y_offset] : input[(i_y + input_height) * input_pitch];
        break;
      case conv_filter_border_type::BORDER_REFLECT_101:   // gfedcb|abcdefgh|gfedcba
        s_Data[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y] = (i_y >= 0) ? input[i_y_offset] : input[-i_y_offset];
        break;
    }
  }

  // Lower halo
  #pragma unroll
  for (int i = HALO_STEPS + RESULT_STEPS, i_y_offset = (i*BLOCKDIM_Y + baseY) * input_pitch; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; i++, i_y_offset += BLOCKDIM_Y * input_pitch) {
    const int i_y = baseY + i * BLOCKDIM_Y;
    switch(BORDER) {
      case conv_filter_border_type::BORDER_SKIP:
        // fall-through
      case conv_filter_border_type::BORDER_ZERO:          // 000000|abcdefgh|0000000
        s_Data[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y]= (input_height > i_y) ? input[i_y_offset] : 0;
        break;
      case conv_filter_border_type::BORDER_REPLICATE:     // aaaaaa|abcdefgh|hhhhhhh
        s_Data[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y]= (input_height > i_y) ? input[i_y_offset] : input[(input_height-1) * input_pitch];
        break;
      case conv_filter_border_type::BORDER_REFLECT:       // fedcba|abcdefgh|hgfedcb
        s_Data[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y]= (input_height > i_y) ? input[i_y_offset] : input[((input_height<<1) - 1 - i_y) * input_pitch];
        break;
      case conv_filter_border_type::BORDER_WRAP:          // cdefgh|abcdefgh|abcdefg
        s_Data[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y]= (input_height > i_y) ? input[i_y_offset] : input[(i_y - input_height) * input_pitch];
        break;
      case conv_filter_border_type::BORDER_REFLECT_101:   // gfedcb|abcdefgh|gfedcba
        s_Data[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y]= (input_height > i_y) ? input[i_y_offset] : input[((input_height<<1) - 2 - i_y) * input_pitch];
        break;
    }
  }

  // Compute and store results
  __syncthreads();
  #pragma unroll
  for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
   float sum = 0;
   #pragma unroll
   for (int j = -RADIUS; j <= RADIUS; j++) {
     sum += filter.d[RADIUS + j] * s_Data[threadIdx.x][threadIdx.y + i * BLOCKDIM_Y + j];
   }
   sum *= scale;
   // Saturate if non-float
   if(sizeof(O) < sizeof(float)) {
     sum = max(min(sum,255.0f),0.f);
   }
   output[i * BLOCKDIM_Y * output_pitch] = sum;
  }
}

template <typename I, typename O>
__host__ void conv_filter_col_gpu(const I * d_image_in,
                                  const int input_pitch,
                                  O * d_image_out,
                                  const int output_pitch,
                                  const int width_px,
                                  const int height_px,
                                  const conv_filter_type_t filter_type,
                                  const conv_filter_border_type_t border_type,
                                  const bool skip_first_and_last_col,
                                  const float scale,
                                  cudaStream_t stream) {
  const filter1x3_t & filter = conv_filter_get1x3(filter_type);

  int width_px_out = width_px - (skip_first_and_last_col?2:0);
  dim3 threads_per_block(BLOCKDIM_X, BLOCKDIM_Y);
  dim3 blocks_per_grid((width_px_out + BLOCKDIM_X -1)/ BLOCKDIM_X,
              (height_px + RESULT_STEPS * BLOCKDIM_Y -1) / (RESULT_STEPS * BLOCKDIM_Y));
  
  // Note: we actually support radiuses up to BLOCKDIM_X * HALO_STEPS, but the filter itself
  //       is not defined beyond 1
  decltype (&conv_filter_col_gpu_shm_kernel<I,O,1,conv_filter_border_type::BORDER_ZERO>) kernel;
  switch(border_type) {
    case conv_filter_border_type::BORDER_SKIP:
    case conv_filter_border_type::BORDER_ZERO:          // 000000|abcdefgh|0000000
      kernel = conv_filter_col_gpu_shm_kernel<I,O,1,conv_filter_border_type::BORDER_ZERO>;
      break;
    case conv_filter_border_type::BORDER_REPLICATE:     // aaaaaa|abcdefgh|hhhhhhh
      kernel = conv_filter_col_gpu_shm_kernel<I,O,1,conv_filter_border_type::BORDER_REPLICATE>;
      break;
    case conv_filter_border_type::BORDER_REFLECT:       // fedcba|abcdefgh|hgfedcb
      kernel = conv_filter_col_gpu_shm_kernel<I,O,1,conv_filter_border_type::BORDER_REFLECT>;
      break;
    case conv_filter_border_type::BORDER_WRAP:          // cdefgh|abcdefgh|abcdefg
      kernel = conv_filter_col_gpu_shm_kernel<I,O,1,conv_filter_border_type::BORDER_WRAP>;
      break;
    case conv_filter_border_type::BORDER_REFLECT_101:   // gfedcb|abcdefgh|gfedcba
      kernel = conv_filter_col_gpu_shm_kernel<I,O,1,conv_filter_border_type::BORDER_REFLECT_101>;
      break;
    default:
      assert(0);
      kernel = conv_filter_col_gpu_shm_kernel<I,O,1,conv_filter_border_type::BORDER_ZERO>;
      break;
  }
  kernel<<<blocks_per_grid,threads_per_block,0,stream>>>(
                                d_image_out + (skip_first_and_last_col?1:0),
                                output_pitch,
                                d_image_in  + (skip_first_and_last_col?1:0),
                                input_pitch,
                                width_px_out,
                                height_px,
                                filter,
                                scale);
  CUDA_KERNEL_CHECK();
}

__host__ void conv_filter_col_cpu(const unsigned char * h_image_in,
                                  const int input_pitch,
                                  unsigned char * h_image_out,
                                  const int output_pitch,
                                  const int width_px,
                                  const int height_px,
                                  const conv_filter_type_t filter_type,
                                  const conv_filter_border_type_t border_type,
                                  const bool skip_first_and_last_col,
                                  const float scale) {
  const filter1x3_t & filter = conv_filter_get1x3(filter_type);
  const int x_min = 0             + (skip_first_and_last_col?1:0);
  const int x_max = (width_px-1)  - (skip_first_and_last_col?1:0);
  const int y_min = 0             + (border_type==conv_filter_border_type::BORDER_SKIP?1:0);
  const int y_max = (height_px-1) - (border_type==conv_filter_border_type::BORDER_SKIP?1:0);
  for(int y=y_min;y<=y_max;++y) {
    for(int x=x_min;x<=x_max;++x) {
      float accu = 0.0f;
      for(int f_y=-1;f_y<=1;++f_y) {
        int i_y = y+f_y;
        switch(border_type) {
          case conv_filter_border_type::BORDER_SKIP:
            // nothing to do
            break;
          case conv_filter_border_type::BORDER_ZERO:        // 000000|abcdefgh|0000000
            // nothing to do
            break;
          case conv_filter_border_type::BORDER_REPLICATE:   // aaaaaa|abcdefgh|hhhhhhh
            i_y = min(max(i_y,0),y_max);
            break;
          case conv_filter_border_type::BORDER_REFLECT:     // fedcba|abcdefgh|hgfedcb
            if(i_y < y_min) {
              i_y = -1*i_y - 1;
            } else if(i_y > y_max) {
              i_y = y_max - (i_y-height_px);
            }
            break;
          case conv_filter_border_type::BORDER_WRAP:        // cdefgh|abcdefgh|abcdefg
            if(i_y < 0) {
              i_y += height_px;
            } else if(i_y > y_max) {
              i_y -= height_px;
            }
            break;
          case conv_filter_border_type::BORDER_REFLECT_101: // gfedcb|abcdefgh|gfedcba
            if(i_y < 0) {
              i_y *= -1;
            } else if(i_y > y_max) {
              i_y = 2*y_max - i_y;
            }
            break;  
        }
        // Handling of BORDER_ZERO
        accu += ((i_y < 0 || i_y >= height_px) ? 0.0f : h_image_in[i_y*input_pitch + x])*filter.d[f_y+1];
      }
      accu *= scale;
      h_image_out[y*output_pitch + x] = static_cast<unsigned char>(min(max(accu,0.0f),255.0f));
    }
  }
}

// Explicit instantiations
INSTANTIATE_1D_COL(unsigned char, unsigned char);
INSTANTIATE_1D_COL(unsigned char, float);
INSTANTIATE_1D_COL(float, unsigned char);
INSTANTIATE_1D_COL(float, float);

} // namespace vilib