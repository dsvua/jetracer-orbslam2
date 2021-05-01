#ifndef JETRACER_CONTEXT_H
#define JETRACER_CONTEXT_H

#include "constants.h"
#include "Ordered.h"
#include <functional>
#include <pthread.h>
#include "Events/EventTypes.h"
#include "Events/BaseEvent.h"

namespace Jetracer
{

    typedef struct
    {
        int cam_w = 848;
        int cam_h = 480;
        int fps = 60; // depth and color realsense streams must support fps for "cam_w x cam_h" resolution
        int PingPong_max_queue_legth = 1;
        int RealSenseD400_autoexposure_settle_frame = 100;
        int RealSenseD400_max_queue_legth = 3;
        int SaveRawData_max_queue_legth = 1;
        int WebSocketCom_max_queue_legth = 1;
        float WebSocketCom_max_send_rate = 5.0f * 1024 * 1024; // ~5Mb/s
        int SlamGpuPipeline_max_queue_length = 5;
        int SlamGpuPipeline_max_streams_length = 1;
        int SlamGpuPipeline_max_keypoints = 1024;
        int SlamGpuPipeline_max_keypoints_to_search = 5120;

        // Fast keypoints
        int pyramid_levels = 1;

        // CUDA
        int CUDA_THREADS_PER_BLOCK = 32;

        // currently unused
        int frames_to_skip = 50; // discard all frames until start_frame to
                                 // give autoexposure, etc. a chance to settle
        int left_gap = 60;       // ignore left 60 pixels on depth image as they
                                 // usually have 0 distance and are useless
        int bottom_gap = 50;     // ignore bottom 50 pixels on depth image
        // unsigned int bottom_gap = 50; // ignore bottom 50 pixels on depth image

        int min_obstacle_height = 5;   // ignore obstacles lower then 5mm
        int max_obstacle_height = 250; // ignore everything higher then 25cm
                                       // as car is not that tall

        Ordered<bool> *stream_video;       // by default do not stream video
        Ordered<bool> *self_drive;         // by default use remote commands
        std::string client_ip_address;     // address of desktop/laptop that controls car
        int websocket_port = 9002;         // port to listen for commands over WebSockets
        int wait_for_thread = 1 * 1000000; // wait for 1 sec for thread to start

        std::function<bool(pEvent)> sendEvent;
        std::function<bool(EventType, std::string, std::function<bool(pEvent)>)> subscribeForEvent;
        std::function<bool(EventType, std::string)> unSubscribeFromEvent;

        // std::string images_path = "/home/serhiy/Downloads/images_temp/";
        std::string images_path = "/workspaces/jetracer-orbslam2/images_temp/";

        // SLAM
        int initial_frame_count = 100;        // number of frames to average distance for keypoints
                                              // when robot is stationary.
        float new_keyframe_angle = 0.003f;    // radians turn for new keyframe
        float new_keyframe_distance = 300.0f; // distance in mm for new keyframe

    } context_t;

} // namespace Jetracer

#endif // JETRACER_CONTEXT_H
