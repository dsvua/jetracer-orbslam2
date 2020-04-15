
#include "image_pipeline.h"
#include <iostream>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API


namespace Jetracer {

    bool imagePipelineThread::threadInitialize() {

        image_size = _ctx->cam_w * _ctx->cam_h;

    }

    bool imagePipelineThread::threadExecute() {


        while (true){
            depth_frame tframe = frame{};
            if (_ctx->depth_queue.poll_for_frame(&tframe))
            {
                // tframe.get_data();
                int image_size
                cv::Mat image = cv::Mat(cv::Size(width, height), CV_16SC1, (void*)tframe.get_data());
                int frame_counter = tframe.get_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER);
                cout << images_path + cv::format("depth_%05u.png", frame_counter) << endl;
                cv::imwrite(images_path + cv::format("depth_%05u.png", frame_counter), image);
            }
        }

        return true;
    }

    bool imagePipelineThread::threadShutdown() {
        return true;
    }

}