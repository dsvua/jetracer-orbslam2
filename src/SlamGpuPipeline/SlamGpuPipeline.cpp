#include "SlamGpuPipeline.h"

#include <memory>
#include <chrono>
#include <iostream>
#include <pthread.h>

#include "../cuda/cuda_RGB_to_Grayscale.cuh"
#include "../cuda/orb.cuh"
#include "../cuda/cuda-align.cuh"
#include "../cuda/pyramid.cuh"
#include "../cuda/fast.cuh"
#include "../cuda/post_processing.cuh"
#include "../cuda_common.h"
#include "defines.h"

#include <unistd.h> // for sleep function
#include <chrono>
#include <nvjpeg.h>
#include <cmath>

using namespace std::chrono;

// using namespace std;

namespace Jetracer
{

    SlamGpuPipeline::SlamGpuPipeline(const std::string threadName, context_t *ctx) : EventsThread(threadName), _ctx(ctx)
    {
        auto pushEventCallback = [this](pEvent event) -> bool {
            this->pushEvent(event);
            return true;
        };

        _ctx->subscribeForEvent(EventType::event_realsense_D400_gyro, threadName, pushEventCallback);
        _ctx->subscribeForEvent(EventType::event_realsense_D400_accel, threadName, pushEventCallback);
        _ctx->subscribeForEvent(EventType::event_realsense_D400_rgbd, threadName, pushEventCallback);
        _ctx->subscribeForEvent(EventType::event_gpu_callback, threadName, pushEventCallback);

        slam_frames = new std::shared_ptr<slam_frame_callback_t>[_ctx->SlamGpuPipeline_max_streams_length];

        for (int i = 0; i < _ctx->SlamGpuPipeline_max_streams_length; i++)
        {
            deleted_slam_frames.push_back(_ctx->SlamGpuPipeline_max_streams_length - i - 1);

            slam_frames[i] = std::make_shared<slam_frame_callback_t>();
            slam_frames[i]->exit_gpu_pipeline = false;
            slam_frames[i]->gpu_thread = std::thread(&SlamGpuPipeline::buildStream, this, i);
        }

        loadPattern(); // loads ORB pattern to GPU
        checkCudaErrors(cudaMalloc((void **)&_d_rgb_intrinsics, sizeof(rs2_intrinsics)));
        checkCudaErrors(cudaMalloc((void **)&_d_depth_intrinsics, sizeof(rs2_intrinsics)));
        checkCudaErrors(cudaMalloc((void **)&_d_depth_rgb_extrinsics, sizeof(rs2_extrinsics)));

        std::cout << "SlamGpuPipeline is initialized" << std::endl;
    }

    void SlamGpuPipeline::upload_intristics(std::shared_ptr<Jetracer::rgbd_frame_t> rgbd_frame)
    {
        // std::cout << "Uploading intinsics " << std::endl;

        // auto rgb_profile = rgbd_frame->rgb_frame.get_profile().as<rs2::video_stream_profile>();
        // auto depth_profile = rgbd_frame->depth_frame.get_profile().as<rs2::video_stream_profile>();

        // rs2_intrinsics h_rgb_intrinsics = rgb_profile.get_intrinsics();
        // rs2_intrinsics h_depth_intrinsics = depth_profile.get_intrinsics();
        // rs2_extrinsics h_depth_rgb_extrinsics = depth_profile.get_extrinsics_to(rgb_profile);

        rs2_intrinsics h_rgb_intrinsics = rgbd_frame->rgb_intristics;
        rs2_intrinsics h_depth_intrinsics = rgbd_frame->depth_intristics;
        rs2_extrinsics h_depth_rgb_extrinsics = rgbd_frame->extrinsics;

        checkCudaErrors(cudaMemcpy((void *)_d_rgb_intrinsics,
                                   &h_rgb_intrinsics,
                                   sizeof(rs2_intrinsics),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void *)_d_depth_intrinsics,
                                   &h_depth_intrinsics,
                                   sizeof(rs2_intrinsics),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void *)_d_depth_rgb_extrinsics,
                                   &h_depth_rgb_extrinsics,
                                   sizeof(rs2_extrinsics),
                                   cudaMemcpyHostToDevice));

        depth_scale = rgbd_frame->depth_scale;
        intristics_are_known = true;
        std::cout << "Uploaded intinsics " << std::endl;
    }

    void SlamGpuPipeline::handleEvent(pEvent event)
    {

        switch (event->event_type)
        {

        case EventType::event_stop_thread:
        {
            std::cout << "Stopping GPU threads" << std::endl;
            for (int i = 0; i < _ctx->SlamGpuPipeline_max_streams_length; i++)
            {
                std::cout << "Stopping GPU thread: " << i << std::endl;
                slam_frames[i]->exit_gpu_pipeline = true;
                slam_frames[i]->thread_cv.notify_one();
                slam_frames[i]->gpu_thread.join();
                // std::cout << "Stopped GPU thread: " << i << std::endl;
            }
            break;
        }

        case EventType::event_realsense_D400_gyro:
        {
            auto imu_frame = std::static_pointer_cast<imu_frame_t>(event->message);
            process_gyro(imu_frame->motion_data, imu_frame->timestamp);
            last_ts_gyro = imu_frame->timestamp;
            break;
        }

        case EventType::event_realsense_D400_accel:
        {
            auto imu_frame = std::static_pointer_cast<imu_frame_t>(event->message);
            process_accel(imu_frame->motion_data);
            break;
        }

        case EventType::event_realsense_D400_rgbd:
        {
            auto rgbd_frame = std::static_pointer_cast<rgbd_frame_t>(event->message);

            // upload intinsics/extrinsics to GPU if not uploaded yet
            if (!intristics_are_known)
            {
                upload_intristics(rgbd_frame);
            }
            else
            {
                rgb_curr_frame_id = rgbd_frame->rgb_frame_id;
                depth_curr_frame_id = rgbd_frame->depth_frame_id;

                // need to check if current frame is not the same as previous
                // as sometimes librealsens sends the same frames few times
                if (deleted_slam_frames.size() > 0 &&
                    frame_counter > _ctx->RealSenseD400_autoexposure_settle_frame &&
                    rgb_curr_frame_id != rgb_prev_frame_id &&
                    depth_curr_frame_id != depth_prev_frame_id)
                {
                    std::lock_guard<std::mutex> lock(m_gpu_mutex);
                    int thread_id = deleted_slam_frames.back();
                    deleted_slam_frames.pop_back();
                    // m_gpu_mutex.unlock();

                    rgbd_frame->theta = theta;
                    slam_frames[thread_id]->rgbd_frame = rgbd_frame;
                    slam_frames[thread_id]->image_ready_for_process = true;
                    slam_frames[thread_id]->thread_cv.notify_one();
                    std::cout << "Notified GPU thread " << thread_id
                              << " rgb frame id: " << rgb_curr_frame_id
                              << " depth frame id: " << depth_curr_frame_id
                              << " GPU queue length: " << _ctx->SlamGpuPipeline_max_streams_length - deleted_slam_frames.size()
                              << std::endl;
                    rgb_prev_frame_id = rgb_curr_frame_id;
                    depth_prev_frame_id = depth_curr_frame_id;
                }
            }
            frame_counter++;
            break;
        }

        default:
        {
            // std::cout << "Got unknown message of type " << event->event_type << std::endl;
            break;
        }
        }
    }

    void SlamGpuPipeline::process_gyro(rs2_vector gyro_data, double ts)
    {
        if (firstGyro) // On the first iteration, use only data from accelerometer to set the camera's initial position
        {
            firstGyro = false;
            last_ts_gyro = ts;
            return;
        }
        // Holds the change in angle, as calculated from gyro
        float3 gyro_angle;

        // Initialize gyro_angle with data from gyro
        gyro_angle.x = gyro_data.x; // Pitch
        gyro_angle.y = gyro_data.y; // Yaw
        gyro_angle.z = gyro_data.z; // Roll

        // Compute the difference between arrival times of previous and current gyro frames
        double dt_gyro = (ts - last_ts_gyro) / 1000.0;
        last_ts_gyro = ts;

        // Change in angle equals gyro measures * time passed since last measurement
        gyro_angle.x *= dt_gyro;
        gyro_angle.y *= dt_gyro;
        gyro_angle.z *= dt_gyro;

        // Apply the calculated change of angle to the current angle (theta)
        theta.x -= gyro_angle.z;
        theta.y -= gyro_angle.y;
        theta.z += gyro_angle.x;
        // theta.add(-gyro_angle.z, -gyro_angle.y, gyro_angle.x);
    }

    void SlamGpuPipeline::process_accel(rs2_vector accel_data)
    {
        // Holds the angle as calculated from accelerometer data
        float3 accel_angle;

        // Calculate rotation angle from accelerometer data
        accel_angle.z = atan2(accel_data.y, accel_data.z);
        accel_angle.x = atan2(accel_data.x, sqrt(accel_data.y * accel_data.y + accel_data.z * accel_data.z));

        // If it is the first iteration, set initial pose of camera according to accelerometer data (note the different handling for Y axis)
        if (firstAccel)
        {
            firstAccel = false;
            theta = accel_angle;
            // Since we can't infer the angle around Y axis using accelerometer data, we'll use PI as a convetion for the initial pose
            theta.y = CUDART_PI_D;
        }
        else
        {
            /* 
            Apply Complementary Filter:
                - high-pass filter = theta * alpha:  allows short-duration signals to pass through while filtering out signals
                  that are steady over time, is used to cancel out drift.
                - low-pass filter = accel * (1- alpha): lets through long term changes, filtering out short term fluctuations 
            */
            theta.x = theta.x * alpha + accel_angle.x * (1 - alpha);
            theta.z = theta.z * alpha + accel_angle.z * (1 - alpha);
        }
    }

} // namespace Jetracer