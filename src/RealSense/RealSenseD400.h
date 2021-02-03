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
        const void *depth = NULL;
        const void *rgb = NULL;
        const void *lefr_ir = NULL;
        const void *right_ir = NULL;
        double timestamp;
        int depth_size;
        int image_size;
        rs2_stream frame_type;
    } rgbd_frame_t;

    typedef struct imu_frame
    {
        rs2_vector motion_data;
        double timestamp;
        rs2_stream frame_type;
    } imu_frame_t;

} // namespace Jetracer

#endif // JETRACER_REALSENSE_D400_THREAD_H
