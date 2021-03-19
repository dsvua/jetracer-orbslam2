#ifndef JETRACER_SLAM_GPU_PIPELINE_THREAD_H
#define JETRACER_SLAM_GPU_PIPELINE_THREAD_H

#include <iostream>

#include "../EventsThread.h"
#include "../Context.h"
#include "../Events/BaseEvent.h"
#include "../Events/EventTypes.h"
#include <mutex>
#include <atomic>
#include <thread>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "../RealSense/RealSenseD400.h"

#define CUDA_WARP_SIZE 32
#define RS2_CUDA_THREADS_PER_BLOCK 32

#define USE_PRECALCULATED_INDICES 1
#define FAST_GPU_USE_LOOKUP_TABLE 1
#define FAST_GPU_USE_LOOKUP_TABLE_BITBASED 1

namespace Jetracer
{

#pragma once

#define CHECK_NVJPEG(call)                                                                                  \
    {                                                                                                       \
        nvjpegStatus_t _e = (call);                                                                         \
        if (_e != NVJPEG_STATUS_SUCCESS)                                                                    \
        {                                                                                                   \
            std::cout << "NVJPEG failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                                                        \
        }                                                                                                   \
    }

    typedef struct slam_frame_callback
    {
        std::shared_ptr<rgbd_frame_t> rgbd_frame;
        bool image_ready_for_process;
        std::thread gpu_thread;
        std::mutex thread_mutex;
        std::condition_variable thread_cv;

    } slam_frame_callback_t;

    typedef struct slam_frame
    {
        unsigned char *image;
        size_t image_length;
        std::shared_ptr<uint16_t[]> keypoints_x;
        std::shared_ptr<uint16_t[]> keypoints_y;
        int keypoints_count;
        std::shared_ptr<rgbd_frame_t> rgbd_frame;

    } slam_frame_t;

    class SlamGpuPipeline : public EventsThread
    {
    public:
        SlamGpuPipeline(const std::string threadName, context_t *ctx);
        // ~SlamGpuPipeline();
        // void pushOverride(pEvent event); // callback events needs to be added no matter what

    private:
        void handleEvent(pEvent event);
        void buildStream(int slam_frames_id);

        void upload_intristics(std::shared_ptr<Jetracer::rgbd_frame_t> rgbd_frame);

        context_t *_ctx;
        std::mutex m_mutex_subscribers;
        int streams_count = 0;
        std::shared_ptr<slam_frame_callback_t> *slam_frames; //shared pointer to auto free memory
        std::vector<int> deleted_slam_frames;
        bool include_anms = false;
        int fastThresh = 20;
        slam_frame_t keyframe;
        unsigned long long rgb_curr_frame_id = 0;
        unsigned long long depth_curr_frame_id = 0;
        unsigned long long rgb_prev_frame_id = 0;
        unsigned long long depth_prev_frame_id = 0;

        bool intristics_are_known = false;
        bool exit_gpu_pipeline = false;
        std::mutex m_gpu_mutex;

        float depth_scale;
        rs2_intrinsics *_d_rgb_intrinsics;
        rs2_intrinsics *_d_depth_intrinsics;
        rs2_extrinsics *_d_depth_rgb_extrinsics;

        int frame_counter = 0;
    };
} // namespace Jetracer

#endif // JETRACER_SLAM_GPU_PIPELINE_THREAD_H
