#ifndef JETRACER_THREAD_EVENT_TYPES_H
#define JETRACER_THREAD_EVENT_TYPES_H

#include <iostream>

namespace Jetracer
{

    // when adding new EventType do not forget to add it
    // in EventTypes.cpp for operator<< overload
    enum class EventType
    {
        // thread events
        event_start_thread,
        event_stop_thread,

        // keep alive
        event_ping,
        event_pong,

        // Realsense D400 events
        event_realsense_D400_rgb,
        event_realsense_D400_rgbd,
        event_realsense_D400_accel,
        event_realsense_D400_gyro,

        // GPU events
        event_gpu_callback,
        event_gpu_slam_frame,
    };

    std::ostream &operator<<(std::ostream &os, EventType &event_type);

} // namespace Jetracer

#endif // JETRACER_THREAD_EVENT_TYPES_H