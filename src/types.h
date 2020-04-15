#include <librealsense2/rs.hpp>
#include "constants.h"
#include <cuda.h>

#ifndef JETRACER_TYPES_THREAD_H
#define JETRACER_TYPES_THREAD_H


namespace Jetracer {

    typedef struct
    {
        unsigned int cam_w = 848;
        unsigned int cam_h = 480;
        unsigned int fps = 6;
        unsigned int frames_to_skip = 30; // discard all frames until start_frame to
                                       // give autoexposure, etc. a chance to settle
        unsigned int left_gap = 60; // ignore left 60 pixels on depth image as they
                                    // usually have 0 distance
        unsigned int bottom_gap = 50; // ignore bottom 50 pixels on depth image

        unsigned int min_obstacle_height = 5; // ignore obstacles lower then 5mm
        unsigned int max_obstacle_height = 250; // ignore everything higher then 25cm
                                        // as car is not that tall

        rs2::frame_queue depth_queue(CAPACITY);
        rs2::frame_queue left_ir_queue(CAPACITY);
        rs2::rs2_intrinsics jetson_camera_intrinsics;

    } context_t;

    // context_t * ctx

} // namespace Jetracer

#endif // JETRACER_TYPES_THREAD_H
