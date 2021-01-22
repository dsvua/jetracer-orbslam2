#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2/hpp/rs_types.hpp>
#include <librealsense2/hpp/rs_frame.hpp>
#include <iostream>             // for cout
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>

using namespace rs2;
using namespace std;

int main(int argc, char * argv[]) {
    int width = 848;
    int height = 480;
    int fps = 15;
    cv::String images_path(argv[1]);

    rs2::config config;
    config.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8, fps);
    config.enable_stream(RS2_STREAM_INFRARED, 2, width, height, RS2_FORMAT_Y8, fps);
    config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);

    // start pipeline
    rs2::pipeline pipeline;
    rs2::pipeline_profile pipeline_profile = pipeline.start(config);

    rs2::device selected_device = pipeline_profile.get_device();
    auto depth_sensor = selected_device.first<rs2::depth_sensor>();
    depth_sensor.set_option(RS2_OPTION_EMITTER_ON_OFF, 1);

    int emitter_mode, frame_count;



    const auto CAPACITY = 5; // allow max latency of 5 frames
    rs2::frame_queue depth_queue(CAPACITY);
    rs2::frame_queue left_ir_queue(CAPACITY);
    rs2::frame_queue right_ir_queue(CAPACITY);


    std::thread depth_thread([&]() {
        while (true)
        {
            depth_frame tframe = frame{};
            if (depth_queue.poll_for_frame(&tframe))
            {
                // tframe.get_data();
                cv::Mat image = cv::Mat(cv::Size(width, height), CV_16SC1, (void*)tframe.get_data());
                int frame_counter = tframe.get_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER);
                cout << images_path + cv::format("depth_%05u.png", frame_counter) << endl;
                cv::imwrite(images_path + cv::format("depth_%05u.png", frame_counter), image);
            }
        }
    });
    depth_thread.detach();

    std::thread left_ir_thread([&]() {
        while (true)
        {
            video_frame tframe = frame{};
            if (left_ir_queue.poll_for_frame(&tframe))
            {
                // tframe.get_data();
                cv::Mat image = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)tframe.get_data());
                int frame_counter = tframe.get_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER);
                cout << images_path + cv::format("left_ir_%05u.png", frame_counter) << endl;
                cv::imwrite(images_path + cv::format("left_ir_%05u.png", frame_counter), image);
            }
        }
    });
    left_ir_thread.detach();

    std::thread right_ir_thread([&]() {
        while (true)
        {
            video_frame tframe = frame{};
            if (right_ir_queue.poll_for_frame(&tframe))
            {
                // tframe.get_data();
                cv::Mat image = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)tframe.get_data());
                int frame_counter = tframe.get_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER);
                cout << images_path + cv::format("right_ir_%05u.png", frame_counter) << endl;
                cv::imwrite(images_path + cv::format("right_ir_%05u.png", frame_counter), image);
            }
        }
    });
    right_ir_thread.detach();

    // Capture 30 frames to give autoexposure, etc. a chance to settle
    for (auto i = 0; i < 30; ++i) pipeline.wait_for_frames();
        
    while (1){

        // wait for frames and get frameset
        rs2::frameset frameset = pipeline.wait_for_frames();

        // get single infrared frame from frameset
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

    return EXIT_SUCCESS;
}