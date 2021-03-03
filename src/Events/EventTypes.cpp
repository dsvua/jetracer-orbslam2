#include "EventTypes.h"

namespace Jetracer
{

    std::ostream &operator<<(std::ostream &os, EventType &event_type)
    {
        switch (event_type)
        {
        case EventType::event_start_thread:
        {
            os << "event_start_thread";
            break;
        }

        case EventType::event_stop_thread:
        {
            os << "event_stop_thread";
            break;
        }

        case EventType::event_ping:
        {
            os << "event_ping";
            break;
        }

        case EventType::event_pong:
        {
            os << "event_pong";
            break;
        }

        case EventType::event_realsense_D400_rgb:
        {
            os << "event_realsense_D400_rgb";
            break;
        }

        case EventType::event_realsense_D400_rgbd:
        {
            os << "event_realsense_D400_rgbd";
            break;
        }

        case EventType::event_realsense_D400_accel:
        {
            os << "event_realsense_D400_accel";
            break;
        }

        case EventType::event_realsense_D400_gyro:
        {
            os << "event_realsense_D400_gyro";
            break;
        }

        case EventType::event_gpu_callback:
        {
            os << "event_gpu_callback";
            break;
        }

        case EventType::event_gpu_slam_frame:
        {
            os << "event_gpu_slam_frame";
            break;
        }

        default:
            break;
        }

        return os;
    }
} // namespace Jetracer