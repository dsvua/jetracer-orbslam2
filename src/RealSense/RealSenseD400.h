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
#include <cuda_runtime.h>
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
        // rs2::depth_frame depth_frame = rs2::frame{};
        // rs2::video_frame rgb_frame = rs2::frame{};
        uint16_t *depth_image;
        unsigned char *rgb_image;
        unsigned char *ir_image;

        float3 theta; // gyro and accel computed angles for this frame

        double timestamp;
        unsigned long long depth_frame_id;
        unsigned long long rgb_frame_id;
        unsigned long long ir_frame_id;

        int depth_image_size;
        int rgb_image_size;
        int ir_image_size;

        rs2_intrinsics depth_intristics;
        rs2_intrinsics rgb_intristics;
        rs2_extrinsics extrinsics;
        float depth_scale;

        // float depth_scale;

        std::chrono::_V2::system_clock::time_point RS400_callback;
        std::chrono::_V2::system_clock::time_point GPU_scheduled;
        std::chrono::_V2::system_clock::time_point GPU_callback;
        std::chrono::_V2::system_clock::time_point GPU_EventSent;

        ~rgbd_frame()
        {
            if (depth_image)
                free(depth_image);
            if (rgb_image)
                free(rgb_image);
            if (ir_image)
                free(ir_image);
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
