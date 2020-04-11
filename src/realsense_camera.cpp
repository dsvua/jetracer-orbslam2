
#include "realsense_camera.h"
#include <iostream>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API


namespace Jetracer {

    bool realsenseD435iThread::threadInitialize() {
        rs2::config config;
        config.enable_stream(RS2_STREAM_INFRARED, 1, _width, _height, RS2_FORMAT_Y8, _fps);
        config.enable_stream(RS2_STREAM_INFRARED, 2, _width, _height, RS2_FORMAT_Y8, _fps);
        config.enable_stream(RS2_STREAM_DEPTH, _width, _height, RS2_FORMAT_Z16, _fps);

        // start pipeline
        rs2::pipeline_profile pipeline_profile = pipeline.start(config);

        rs2::device selected_device = pipeline_profile.get_device();
        depth_sensor = selected_device.first<rs2::depth_sensor>();
        depth_sensor.set_option(RS2_OPTION_EMITTER_ON_OFF, 1);

    }

    bool realsenseD435iThread::threadExecute() {

        // Capture 30 frames to give autoexposure, etc. a chance to settle
        for (auto i = 0; i < 30; ++i) pipeline.wait_for_frames();

        while (true){

            // wait for frames and get frameset
            rs2::frameset frameset = pipeline.wait_for_frames();

            // get single frame from frameset
            rs2::video_frame depth_frame = frameset.get_depth_frame();
            emitter_mode = depth_frame.get_frame_metadata(RS2_FRAME_METADATA_FRAME_EMITTER_MODE);
            frame_count = depth_frame.get_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER);
            cout << "Emitter mode: " << emitter_mode << "frame count: " << frame_count << endl;


            if (!emitter_mode) {
                depth_queue.enqueue(frameset.get_depth_frame());
            } else {
                left_ir_queue.enqueue(frameset.get_infrared_frame(1));
                right_ir_queue.enqueue(frameset.get_infrared_frame(2));            
            }
        }

        return true;
    }

    bool realsenseD435iThread::threadShutdown() {
        return true;
    }

}