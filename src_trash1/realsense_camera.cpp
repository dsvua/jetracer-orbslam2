
#include "realsense_camera.h"
#include "types.h"
#include <iostream>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API


namespace Jetracer {

    bool realsenseD435iThread::threadInitialize() {
        _config.enable_stream(RS2_STREAM_DEPTH, _ctx->cam_w, _ctx->cam_h, RS2_FORMAT_Z16, _ctx->fps);
        _config.enable_stream(RS2_STREAM_INFRARED, 1, _ctx->cam_w, _ctx->cam_h, RS2_FORMAT_Y8, _ctx->fps);
        // config.enable_stream(RS2_STREAM_INFRARED, 2, _ctx->cam_w, _ctx->cam_h, RS2_FORMAT_Y8, _ctx->fps);

        // start pipeline
        _pipeline_profile = _pipeline.start(_config);

        // rs2::device selected_device = pipeline_profile.get_device();
        // rs2::depth_sensor depth_sensor = selected_device.first<rs2::depth_sensor>();
        // depth_sensor.set_option(RS2_OPTION_EMITTER_ON_OFF, 1);

    }

    bool realsenseD435iThread::threadExecute() {

        // Capture 30 frames to give autoexposure, etc. a chance to settle
        for (auto i = 0; i < frames_to_skip; ++i) _pipeline.wait_for_frames();

        while (true){

            // wait for frames and get frameset
            rs2::frameset frameset = _pipeline.wait_for_frames();

            // get single frame from frameset
            rs2::depth_frame depth_frame = frameset.get_depth_frame();
            // int emitter_mode, frame_count;
            // frame_count = depth_frame.get_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER);
            // cout << "Emitter mode: " << emitter_mode << "frame count: " << frame_count << endl;

            _ctx->depth_queue.enqueue(frameset.get_depth_frame());
            _ctx->left_ir_queue.enqueue(frameset.get_infrared_frame(1));
        }

        return true;
    }

    bool realsenseD435iThread::threadShutdown() {
        return true;
    }

}