/*
 * TimerGPU class for profiling algorithms on the GPU
 * timergpu.h
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

#pragma once

#include <cuda.h>
#include "statistics.h"

namespace vilib
{

  class TimerGPU
  {
  public:
    TimerGPU(const char *name, int indent = 1);
    TimerGPU(const std::string &name, int indent = 1);
    ~TimerGPU(void);

    void start(cudaStream_t stream = 0);
    void stop(bool synchronize = true, cudaStream_t stream = 0);
    void sync(void);
    void add_to_stat_n_reset(void);
    double elapsed_usec(void) const { return time_; }
    double elapsed_sec(void) const { return time_ / 1.0e6; }

    void display_usec(void) const;
    void display_stat_usec(void) const;

  private:
    std::string name_;
    double time_; // time is expressed in usec
    int indent_;
    Statistics stat_;
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;

    static const int name_size_characters_ = 35;
    static const int time_size_characters_ = 15;
  };

} // namespace vilib
