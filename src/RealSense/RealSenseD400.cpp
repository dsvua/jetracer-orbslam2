#include "RealSenseD400.h"

#include <memory>
#include <chrono>
#include <iostream>
#include <algorithm> // for std::find_if

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
                for (const rs2::frame &f : fs)
                {
                    // std::cout << " " << f.get_profile().stream_name(); // will print: Depth Infrared 1
                }
                // std::cout << std::endl;

                rgbd_frame_t rgbd_frame;
                rgbd_frame.depth = fs.get_depth_frame().get_data();
                rgbd_frame.lefr_ir = fs.get_infrared_frame().get_data();
                rgbd_frame.timestamp = fs.get_depth_frame().get_timestamp();
                rgbd_frame.depth_size = fs.get_depth_frame().get_data_size();
                rgbd_frame.image_size = fs.get_infrared_frame().get_data_size();
                rgbd_frame.frame_type = RS2_STREAM_INFRARED;

                event->event_type = EventType::event_realsense_D400_rgbd;
                event->message = std::make_shared<rgbd_frame_t>(rgbd_frame);
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
                // std::cout << " " << frame.get_profile().stream_name();
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
                    imu_frame.frame_type = RS2_STREAM_GYRO;

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

        // auto pushEventCallback = [this](pEvent event) -> bool {
        //     this->pushEvent(event);
        //     return true;
        // };

        // _ctx->subscribeForEvent(EventType::event_ping, threadName, pushEventCallback);
        // _ctx->subscribeForEvent(EventType::event_pong, threadName, pushEventCallback);

        //Add desired streams to configuration
        cfg.enable_stream(RS2_STREAM_INFRARED, 1, _ctx->cam_w, _ctx->cam_h, RS2_FORMAT_Y8, _ctx->fps); // fps for 848x480: 30, 60, 90
        // cfg.enable_stream(RS2_STREAM_COLOR, _ctx->cam_w, _ctx->cam_h, RS2_FORMAT_RGB8, 60);
        cfg.enable_stream(RS2_STREAM_DEPTH, _ctx->cam_w, _ctx->cam_h, RS2_FORMAT_Z16, _ctx->fps);
        cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 250); // 63 and 250
        cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200);  // 200 and 400

        // Start the camera pipeline
        selection = pipe.start(cfg, callbackNewFrame);

        // disabling laser
        rs2::device selected_device = selection.get_device();
        auto depth_sensor = selected_device.first<rs2::depth_sensor>();
        depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f); // Disable emitter

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
            pipe.stop();
            break;
        }

        default:
        {
            std::cout << "Got unknown message of type " << event->event_type << std::endl;
            break;
        }
        }
    }

    rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile> &streams)
    {
        //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
        //We prioritize color streams to make the view look better.
        //If color is not available, we take another stream that (other than depth)
        rs2_stream align_to = RS2_STREAM_ANY;
        bool depth_stream_found = false;
        bool color_stream_found = false;
        for (rs2::stream_profile sp : streams)
        {
            rs2_stream profile_stream = sp.stream_type();
            if (profile_stream != RS2_STREAM_DEPTH)
            {
                if (!color_stream_found) //Prefer color
                    align_to = profile_stream;

                if (profile_stream == RS2_STREAM_COLOR)
                {
                    color_stream_found = true;
                }
            }
            else
            {
                depth_stream_found = true;
            }
        }

        if (!depth_stream_found)
            throw std::runtime_error("No Depth stream available");

        if (align_to == RS2_STREAM_ANY)
            throw std::runtime_error("No stream found to align with Depth");

        return align_to;
    }

    bool profile_changed(const std::vector<rs2::stream_profile> &current, const std::vector<rs2::stream_profile> &prev)
    {
        for (auto &&sp : prev)
        {
            //If previous profile is in current (maybe just added another)
            auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile &current_sp) { return sp.unique_id() == current_sp.unique_id(); });
            if (itr == std::end(current)) //If it previous stream wasn't found in current
            {
                return true;
            }
        }
        return false;
    }

} // namespace Jetracer