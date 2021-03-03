#ifndef JETRACER_REALSENSE_D400_THREAD_H
#define JETRACER_REALSENSE_D400_THREAD_H

#include <iostream>

#include "../EventsThread.h"
#include "../Context.h"
#include "../Events/BaseEvent.h"
#include "../Events/EventTypes.h"
#include <mutex>
#include <atomic>
#include <thread>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/core/cuda.hpp>
#include <chrono>

namespace Jetracer
{

    class RealSenseD400 : public EventsThread
    {
    public:
        RealSenseD400(const std::string threadName, context_t *ctx);
        // ~RealSenseD400();

    private:
        void handleEvent(pEvent event);

        context_t *_ctx;
        std::mutex m_mutex_subscribers;

        rs2_intrinsics intrinsics;
        rs2::config cfg;
        rs2::pipeline pipe;
        rs2::pipeline_profile selection;
    };

    typedef struct rgbd_frame
    {
        // cv::cuda::GpuMat d_depth_image;
        // cv::cuda::GpuMat d_rgb_image;
        rs2::depth_frame depth_frame = rs2::frame{};
        rs2::video_frame rgb_frame = rs2::frame{};

        // double timestamp;
        // unsigned long long frame_id;

        // rs2_intrinsics depth_intristics;
        // rs2_intrinsics rgb_intristics;
        // rs2_extrinsics extrinsics;

        // float depth_scale;

        std::chrono::_V2::system_clock::time_point RS400_callback;
        std::chrono::_V2::system_clock::time_point GPU_scheduled;
        std::chrono::_V2::system_clock::time_point GPU_callback;
        std::chrono::_V2::system_clock::time_point GPU_EventSent;

        ~rgbd_frame()
        {
            // d_depth_image.~GpuMat();
            // d_rgb_image.~GpuMat();
        }

    } rgbd_frame_t;

    typedef struct imu_frame
    {
        rs2_vector motion_data;
        double timestamp;
        rs2_stream frame_type;
    } imu_frame_t;

} // namespace Jetracer

#endif // JETRACER_REALSENSE_D400_THREAD_H
