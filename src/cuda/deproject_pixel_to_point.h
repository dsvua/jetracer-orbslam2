#include <librealsense2/rs.hpp>

#ifndef JETRACER_DEPROJECT_PIXEL_TO_POINT_H
#define JETRACER_DEPROJECT_PIXEL_TO_POINT_H


namespace Jetracer {

    typedef struct {
        unsigned int width;
        unsigned int height;
        unsigned int left_gap;
        unsigned int bottom_gap;
        unsigned int min_obstacle_height;
        unsigned int max_obstacle_height;

    } d_context_t;

    void deproject_pixel_to_point(context_t* ctx, rs2::depth_frame depth_frame);
} // namespace Jetracer

#endif // JETRACER_DEPROJECT_PIXEL_TO_POINT_H