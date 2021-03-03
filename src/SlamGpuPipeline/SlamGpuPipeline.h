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

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <cuda_runtime.h>
#include <helper_cuda.h>
// #include <cuda/Cuda.hpp>
// #include "../external/vilib/feature_detection/detector_base_gpu.h"

#include "../RealSense/RealSenseD400.h"

#define CUDA_WARP_SIZE 32
#define RS2_CUDA_THREADS_PER_BLOCK 32

#define USE_PRECALCULATED_INDICES 1
#define FAST_GPU_USE_LOOKUP_TABLE 1
#define FAST_GPU_USE_LOOKUP_TABLE_BITBASED 1

namespace Jetracer
{
    using namespace cv;
    using namespace cv::cuda;

#pragma once
    // enum fast_score
    // {
    //     SUM_OF_ABS_DIFF_ALL = 0, // OpenCV: https://docs.opencv.org/master/df/d0c/tutorial_py_fast.html
    //     SUM_OF_ABS_DIFF_ON_ARC,  // Rosten 2006
    //     MAX_THRESHOLD            // Rosten 2008
    // };

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
        cv::cuda::GpuMat d_keypoints;
        cv::cuda::GpuMat d_descriptors;
        // cv::cuda::GpuMat d_matches;

        cv::Mat descriptors;
        cv::Mat image;
        std::vector<cv::DMatch> feature_matches;
        // std::shared_ptr<std::vector<vilib::DetectorBase<true>::FeaturePoint>> keypoints;
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
        std::shared_ptr<rs2_intrinsics> _d_rgb_intrinsics;
        std::shared_ptr<rs2_intrinsics> _d_depth_intrinsics;
        std::shared_ptr<rs2_extrinsics> _d_depth_rgb_extrinsics;

        int frame_counter = 0;
    };
} // namespace Jetracer

#endif // JETRACER_SLAM_GPU_PIPELINE_THREAD_H
