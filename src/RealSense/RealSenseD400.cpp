#include "RealSenseD400.h"

#include <memory>
#include <chrono>
#include <iostream>
#include <algorithm> // for std::find_if
#include <cstring>
#include <librealsense2/rs_advanced_mode.hpp>

// using namespace std;

namespace Jetracer
{
    float get_depth_scale(rs2::device dev);
    rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile> &streams);
    bool profile_changed(const std::vector<rs2::stream_profile> &current, const std::vector<rs2::stream_profile> &prev);

    RealSenseD400::RealSenseD400(const std::string threadName, context_t *ctx) : EventsThread(threadName), _ctx(ctx)
    {

        // callback for new frames as here: https://github.com/IntelRealSense/librealsense/blob/master/examples/callback/rs-callback.cpp
        auto callbackNewFrame = [this](const rs2::frame &frame) {
            rs2::frameset fs = frame.as<rs2::frameset>();
            pEvent event = std::make_shared<BaseEvent>();

            if (fs)
            {
                // With callbacks, all synchronized streams will arrive in a single frameset
                for (auto &&fr : fs)
                {
                    std::cout << " " << fr.get_profile().stream_name(); // will print: Depth Infrared 1
                }

                auto rs_depth_frame = fs.get_depth_frame();
                auto rs_rgb_frame = fs.get_color_frame();
                auto rs_ir_frame = fs.get_infrared_frame();
                auto image_height = rs_depth_frame.get_height();
                auto image_width = rs_depth_frame.get_width();

                std::cout << " frame id: " << rs_depth_frame.get_frame_number()
                          << " " << rs_ir_frame.get_frame_number()
                          << " " << rs_rgb_frame.get_frame_number() << std::endl;

                // std::cout << " frame id: " << fs.get_depth_frame().get_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER) << " "
                //           << fs.get_color_frame().get_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER) << std::endl;

                // fs.keep();
                // rs_depth_frame.keep();
                // rs_rgb_frame.keep();
                // rs_ir_frame.keep();

                auto rgbd_frame = std::make_shared<rgbd_frame_t>();

                // rgbd_frame->depth_frame = rs_depth_frame;
                // rgbd_frame->rgb_frame = rs_rgb_frame;

                rgbd_frame->timestamp = rs_depth_frame.get_frame_metadata(RS2_FRAME_METADATA_BACKEND_TIMESTAMP);
                // rgbd_frame->depth_frame_id = rs_depth_frame.get_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER);
                // rgbd_frame->rgb_frame_id = rs_rgb_frame.get_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER);
                // rgbd_frame->ir_frame_id = rs_ir_frame.get_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER);
                rgbd_frame->depth_frame_id = rs_depth_frame.get_frame_number();
                rgbd_frame->rgb_frame_id = rs_rgb_frame.get_frame_number();
                rgbd_frame->ir_frame_id = rs_ir_frame.get_frame_number();

                auto depth_profile = rs_depth_frame.get_profile().as<rs2::video_stream_profile>();
                auto rgb_profile = rs_rgb_frame.get_profile().as<rs2::video_stream_profile>();

                rgbd_frame->depth_intristics = depth_profile.get_intrinsics();
                rgbd_frame->rgb_intristics = rgb_profile.get_intrinsics();
                rgbd_frame->extrinsics = depth_profile.get_extrinsics_to(rgb_profile);

                rgbd_frame->depth_scale = rs_depth_frame.get_units();

                // rgbd_frame->original_frame = fs;
                // rgbd_frame->depth = fs.get_depth_frame().get_data();
                // rgbd_frame->rgb = fs.get_color_frame().get_data();
                // rgbd_frame->lefr_ir = fs.get_infrared_frame().get_data();
                // rgbd_frame->depth_size = fs.get_depth_frame().get_data_size();
                // rgbd_frame->image_size = fs.get_color_frame().get_data_size();
                // rgbd_frame->frame_type = RS2_STREAM_COLOR;
                // rgbd_frame->frame_type = RS2_STREAM_INFRARED;
                // rgbd_frame->RS400_callback = std::chrono::high_resolution_clock::now();

                rgbd_frame->depth_image_size = rs_depth_frame.get_data_size();
                rgbd_frame->rgb_image_size = rs_rgb_frame.get_data_size();
                rgbd_frame->ir_image_size = rs_ir_frame.get_data_size();

                rgbd_frame->depth_image = (uint16_t *)malloc(rgbd_frame->depth_image_size * sizeof(char));
                rgbd_frame->rgb_image = (unsigned char *)malloc(rgbd_frame->rgb_image_size * sizeof(char));
                rgbd_frame->ir_image = (unsigned char *)malloc(rgbd_frame->ir_image_size * sizeof(char));

                std::memcpy(rgbd_frame->depth_image, rs_depth_frame.get_data(), rgbd_frame->depth_image_size);
                std::memcpy(rgbd_frame->rgb_image, rs_rgb_frame.get_data(), rgbd_frame->rgb_image_size);
                std::memcpy(rgbd_frame->ir_image, rs_ir_frame.get_data(), rgbd_frame->ir_image_size);

                event->event_type = EventType::event_realsense_D400_rgbd;
                event->message = rgbd_frame;
                this->_ctx->sendEvent(event);

                // sending RGB color image
                // pEvent video_event = std::make_shared<BaseEvent>();
                // rgb_frame_t rgb_frame;
                // rgb_frame.image = fs.get_color_frame().get_data();
                // rgb_frame.timestamp = fs.get_color_frame().get_timestamp();
                // rgb_frame.image_size = fs.get_color_frame().get_data_size();

                // video_event->event_type = EventType::event_realsense_D400_rgb;
                // video_event->message = std::make_shared<rgb_frame_t>(rgb_frame);
                // this->_ctx->sendEvent(video_event);
            }
            else
            {
                // std::cout << " " << frame.get_profile().stream_name() << std::endl;
                switch (frame.get_profile().stream_type())
                {
                case RS2_STREAM_GYRO:
                {
                    auto motion = frame.as<rs2::motion_frame>();

                    imu_frame_t imu_frame;
                    imu_frame.motion_data = motion.get_motion_data();
                    imu_frame.timestamp = motion.get_timestamp();
                    imu_frame.frame_type = RS2_STREAM_GYRO;

                    event->event_type = EventType::event_realsense_D400_gyro;
                    event->message = std::make_shared<imu_frame_t>(imu_frame);

                    this->_ctx->sendEvent(event);
                    break;
                }

                case RS2_STREAM_ACCEL:
                {
                    auto motion = frame.as<rs2::motion_frame>();

                    imu_frame_t imu_frame;
                    imu_frame.motion_data = motion.get_motion_data();
                    imu_frame.timestamp = motion.get_timestamp();
                    imu_frame.frame_type = RS2_STREAM_ACCEL;

                    event->event_type = EventType::event_realsense_D400_accel;
                    event->message = std::make_shared<imu_frame_t>(imu_frame);

                    this->_ctx->sendEvent(event);
                    break;
                }

                default:
                    break;
                }
            }

            // std::cout << std::endl;
        };

        auto pushEventCallback = [this](pEvent event) -> bool {
            this->pushEvent(event);
            return true;
        };

        _ctx->subscribeForEvent(EventType::event_stop_thread, threadName, pushEventCallback);
        // _ctx->subscribeForEvent(EventType::event_ping, threadName, pushEventCallback);
        // _ctx->subscribeForEvent(EventType::event_pong, threadName, pushEventCallback);

        //Add desired streams to configuration
        cfg.enable_stream(RS2_STREAM_INFRARED, 1, _ctx->cam_w, _ctx->cam_h, RS2_FORMAT_Y8, _ctx->fps); // fps for 848x480: 30, 60, 90
        cfg.enable_stream(RS2_STREAM_COLOR, _ctx->cam_w, _ctx->cam_h, RS2_FORMAT_RGB8, _ctx->fps);
        cfg.enable_stream(RS2_STREAM_DEPTH, _ctx->cam_w, _ctx->cam_h, RS2_FORMAT_Z16, _ctx->fps);
        cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 63); // 63 or 250
        cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200); // 200 or 400

        // Start the camera pipeline
        // selection = pipe.start(cfg, callbackNewFrame);
        // selection = pipe.start(cfg);

        // for (int i = 0; i < _ctx->RealSenseD400_autoexposure_settle_frame; i++)
        //     pipe.wait_for_frames();

        // disabling laser
        // rs2::device selected_device = selection.get_device();
        // auto depth_sensor = selected_device.first<rs2::depth_sensor>();
        // auto color_sensor = selected_device.first<rs2::color_sensor>();

        // auto advanced_mode_depth = depth_sensor.as<rs400::advanced_mode>();
        // auto STAEControl_depth = advanced_mode_depth.get_ae_control();
        // STAEControl_depth.meanIntensitySetPoint = 2300;
        // advanced_mode_depth.set_ae_control(STAEControl_depth);

        // depth_sensor.set_option(RS2_OPTION_FRAMES_QUEUE_SIZE, 0);
        // color_sensor.set_option(RS2_OPTION_FRAMES_QUEUE_SIZE, 0);
        // depth_sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0);
        // color_sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0);
        // depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f); // Disable emitter
        // day
        // depth_sensor.set_option(RS2_OPTION_EXPOSURE, 5715.f);     // Change exposure for D455
        // depth_sensor.set_option(RS2_OPTION_GAIN, 41.f);           // Change gain for D455
        // night
        // depth_sensor.set_option(RS2_OPTION_EXPOSURE, 24286.f); // Change exposure for D455
        // depth_sensor.set_option(RS2_OPTION_GAIN, 85.f);        // Change gain for D455

        // Each depth camera might have different units for depth pixels, so we get it here
        // Using the pipeline's profile, we can retrieve the device that the pipeline uses
        // float depth_scale = get_depth_scale(selection.get_device());

        // Pipeline could choose a device that does not have a color stream
        // If there is no color stream, choose to align depth to another stream
        // auto align_to = find_stream_to_align(selection.get_streams());

        // Create a rs2::align object.
        // rs2::align allows us to perform alignment of depth frames to others frames
        // The "align_to" is the stream type to which we plan to align depth frames.
        // auto align(align_to);

        // pipe.stop();
        selection = pipe.start(cfg, callbackNewFrame);

        // Get camera intrinsics
        auto depth_stream = selection.get_stream(RS2_STREAM_DEPTH)
                                .as<rs2::video_stream_profile>();

        // auto resolution = std::make_pair(depth_stream.width(), depth_stream.height());
        intrinsics = depth_stream.get_intrinsics();
        // auto principal_point = std::make_pair(i.ppx, i.ppy);
        // auto focal_length = std::make_pair(i.fx, i.fy);

        std::cout << "ppx: " << intrinsics.ppx << " ppy: " << intrinsics.ppy << std::endl;
        std::cout << "fx: " << intrinsics.fx << " fy: " << intrinsics.fy << std::endl;
        std::cout << "k1: " << intrinsics.coeffs[0] << " k2: " << intrinsics.coeffs[1] << " p1: " << intrinsics.coeffs[2] << " p2: " << intrinsics.coeffs[3] << " k3: " << intrinsics.coeffs[4] << std::endl;

        std::cout << "RealSenseD400 is initialized" << std::endl;
    }

    void RealSenseD400::handleEvent(pEvent event)
    {

        switch (event->event_type)
        {

        case EventType::event_stop_thread:
        {
            std::cout << "Stopping Realsense pipeline " << std::endl;
            pipe.stop();
            std::cout << "Stopped Realsense pipeline " << std::endl;
            break;
        }

        default:
        {
            std::cout << "Got unknown message of type " << event->event_type << std::endl;
            break;
        }
        }
    }

} // namespace Jetracer