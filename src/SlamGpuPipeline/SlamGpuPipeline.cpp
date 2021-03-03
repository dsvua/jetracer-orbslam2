#include "SlamGpuPipeline.h"

#include <memory>
#include <chrono>
#include <iostream>
#include <pthread.h>
#include <opencv2/cudaimgproc.hpp>
#include "../external/vilib/preprocess/pyramid.h"
#include "../external/vilib/storage/pyramid_pool.h"
// #include "../external/vilib/feature_detection/fast/fast_common.h"
#include "../external/vilib/feature_detection/fast/fast_gpu.h"
// #include "../external/vilib/config.h"
// #include "vilib/common/subframe.h"

#include "../cuda/rscuda_utils.cuh"
#include "../cuda/cuda_RGB_to_Grayscale.cuh"
#include "../cuda/orb.cuh"
#include "defines.h"

#include <unistd.h> // for sleep function
#include <chrono>
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

        _ctx->subscribeForEvent(EventType::event_realsense_D400_rgbd, threadName, pushEventCallback);
        _ctx->subscribeForEvent(EventType::event_gpu_callback, threadName, pushEventCallback);

        slam_frames = new std::shared_ptr<slam_frame_callback_t>[_ctx->SlamGpuPipeline_max_streams_length];

        for (int i = 0; i < _ctx->SlamGpuPipeline_max_streams_length; i++)
        {
            deleted_slam_frames.push_back(_ctx->SlamGpuPipeline_max_streams_length - i - 1);

            slam_frames[i] = std::make_shared<slam_frame_callback_t>();
            slam_frames[i]->gpu_thread = std::thread(&SlamGpuPipeline::buildStream, this, i);
        }

        loadPattern(); // loads ORB pattern to GPU

        std::cout << "SlamGpuPipeline is initialized" << std::endl;
    }

    void SlamGpuPipeline::upload_intristics(std::shared_ptr<Jetracer::rgbd_frame_t> rgbd_frame)
    {
        std::cout << "Uploading intinsics " << std::endl;

        auto rgb_profile = rgbd_frame->rgb_frame.get_profile().as<rs2::video_stream_profile>();
        auto depth_profile = rgbd_frame->depth_frame.get_profile().as<rs2::video_stream_profile>();

        _d_rgb_intrinsics = rscuda::make_device_copy(rgb_profile.get_intrinsics());
        _d_depth_intrinsics = rscuda::make_device_copy(depth_profile.get_intrinsics());
        _d_depth_rgb_extrinsics = rscuda::make_device_copy(depth_profile.get_extrinsics_to(rgb_profile));

        depth_scale = rgbd_frame->depth_frame.get_units();
        intristics_are_known = true;
        std::cout << "Uploaded intinsics " << std::endl;
    }

    void SlamGpuPipeline::buildStream(int slam_frames_id)
    {
        // spread threads between cores
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(slam_frames_id + 1, &cpuset);

        int rc = pthread_setaffinity_np(slam_frames[slam_frames_id]->gpu_thread.native_handle(),
                                        sizeof(cpu_set_t), &cpuset);
        if (rc != 0)
        {
            std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
        }

        cudaStream_t stream;
        cudaStream_t io_stream;
        checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        // allocate memory for image processing
        unsigned char *d_rgb_image;
        unsigned char *d_gray_image;
        float *d_gray_keypoint_response;
        bool *d_keypoints_exist;
        float *d_keypoints_response;
        float *d_keypoints_angle;
        float2 *d_keypoints_pos;
        uchar *d_descriptors;

        int grid_cols = (_ctx->cam_w + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
        int grid_rows = (_ctx->cam_h + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
        int keypoints_num = grid_cols * grid_rows;

        int data_bytes = _ctx->cam_w * _ctx->cam_h * 3;
        std::size_t width_char = sizeof(char) * _ctx->cam_w;
        std::size_t width_float = sizeof(float) * _ctx->cam_w;
        std::size_t height = _ctx->cam_h;

        std::size_t rgb_pitch;
        std::size_t gray_pitch;
        std::size_t gray_response_pitch;
        checkCudaErrors(cudaMallocPitch((void **)&d_rgb_image, &rgb_pitch, width_char * 3, height));
        checkCudaErrors(cudaMallocPitch((void **)&d_gray_image, &gray_pitch, width_char, height));
        checkCudaErrors(cudaMallocPitch((void **)&d_gray_image_blurred, &gray_pitch, width_char, height));
        checkCudaErrors(cudaMalloc((void **)&d_keypoints_exist, keypoints_num * sizeof(bool)));
        checkCudaErrors(cudaMalloc((void **)&d_keypoints_response, keypoints_num * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_keypoints_angle, keypoints_num * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_keypoints_pos, keypoints_num * sizeof(float2)));
        checkCudaErrors(cudaMalloc((void **)&d_descriptors, keypoints_num * 32 * sizeof(char)));

        float2 *h_keypoints_pos = (float2 *)malloc(keypoints_num * sizeof(float2));

        std::cout
            << "GPU thread " << slam_frames_id << " is started on CPU: "
            << sched_getcpu() << std::endl;

        std::shared_ptr<uint16_t[]> keypoints_x(new uint16_t[keypoints_num]);
        std::shared_ptr<uint16_t[]> keypoints_y(new uint16_t[keypoints_num]);

        std::shared_ptr<vilib::DetectorBaseGPU> detector_gpu_;
        detector_gpu_.reset(new vilib::FASTGPU(std::size_t(_ctx->cam_w),
                                               std::size_t(_ctx->cam_h),
                                               std::size_t(CELL_SIZE_WIDTH),
                                               std::size_t(CELL_SIZE_HEIGHT),
                                               std::size_t(PYRAMID_MIN_LEVEL),
                                               std::size_t(PYRAMID_MAX_LEVEL),
                                               std::size_t(HORIZONTAL_BORDER),
                                               std::size_t(VERTICAL_BORDER),
                                               FAST_EPSILON,
                                               FAST_MIN_ARC_LENGTH,
                                               vilib::fast_score(FAST_SCORE)));
        detector_gpu_->setStream(stream);

        std::vector<std::shared_ptr<vilib::Subframe>> pyramid_;

        // Initialize the pyramid pool
        vilib::PyramidPool::init(slam_frames_id,
                                 std::size_t(_ctx->cam_w),
                                 std::size_t(_ctx->cam_h),
                                 1, // grayscale
                                 PYRAMID_LEVELS,
                                 SLAM_IMAGE_PYRAMID_MEMORY_TYPE);

        vilib::PyramidPool::get(slam_frames_id,
                                std::size_t(_ctx->cam_w),
                                std::size_t(_ctx->cam_h),
                                sizeof(char),
                                PYRAMID_LEVELS,
                                SLAM_IMAGE_PYRAMID_MEMORY_TYPE,
                                pyramid_);

        std::cout << " vilib::PyramidPool initialized " << std::endl;

        while (!exit_gpu_pipeline)
        {
            std::unique_lock<std::mutex> lk(slam_frames[slam_frames_id]->thread_mutex);
            while (!slam_frames[slam_frames_id]->image_ready_for_process)
                slam_frames[slam_frames_id]->thread_cv.wait(lk);

            if (!slam_frames[slam_frames_id]->image_ready_for_process)
                continue;

            auto slam_frame = std::make_shared<slam_frame_t>();
            std::chrono::_V2::system_clock::time_point stop;
            std::chrono::_V2::system_clock::time_point start = high_resolution_clock::now();
            cv::Mat image(_ctx->cam_h, _ctx->cam_w, CV_8UC1);

            checkCudaErrors(cudaMemcpy2DAsync((void *)d_rgb_image,
                                              rgb_pitch,
                                              slam_frames[slam_frames_id]->rgbd_frame->rgb_frame.get_data(),
                                              _ctx->cam_w * 3,
                                              _ctx->cam_w * 3,
                                              _ctx->cam_h,
                                              cudaMemcpyHostToDevice,
                                              stream));

            // rgb_to_grayscale(pyramid_[0]->data_,
            //                  d_rgb_image,
            //                  _ctx->cam_w,
            //                  _ctx->cam_h,
            //                  pyramid_[0]->pitch_,
            //                  rgb_pitch,
            //                  stream);

            rgb_to_grayscale(d_gray_image,
                             d_rgb_image,
                             _ctx->cam_w,
                             _ctx->cam_h,
                             gray_pitch,
                             rgb_pitch,
                             stream);

            gaussian_blur_3x3(pyramid_[0]->data_,
                              pyramid_[0]->pitch_,
                              d_gray_image,
                              gray_pitch,
                              _ctx->cam_w,
                              _ctx->cam_h,
                              stream);

            checkCudaErrors(cudaMemcpy2DAsync((void *)image.data,
                                              image.step,
                                              pyramid_[0]->data_,
                                              pyramid_[0]->pitch_,
                                              _ctx->cam_w,
                                              _ctx->cam_h,
                                              cudaMemcpyDeviceToHost,
                                              stream));

            vilib::pyramid_create_gpu(pyramid_, stream);

            detector_gpu_->detect(pyramid_);

            compute_fast_angle(d_keypoints_angle,
                               detector_gpu_->d_pos_,
                               pyramid_[0]->data_,
                               pyramid_[0]->pitch_,
                               _ctx->cam_w,
                               _ctx->cam_h,
                               keypoints_num,
                               stream);

            calc_orb(d_keypoints_angle,
                     detector_gpu_->d_pos_,
                     d_descriptors,
                     pyramid_[0]->data_,
                     pyramid_[0]->pitch_,
                     _ctx->cam_w,
                     _ctx->cam_h,
                     keypoints_num,
                     stream);

            stop = high_resolution_clock::now();
            auto &points_gpu = detector_gpu_->getPoints();

#pragma unroll
            // copy keypoints to slam_frame
            for (int i = 0; i < keypoints_num; i++)
            {
                keypoints_x[i] = uint16_t(points_gpu[i].x_);
                keypoints_y[i] = uint16_t(points_gpu[i].y_);
            }

            // sleep(2);
            // std::cout << "cudaStreamSynchronize" << std::endl;
            // checkCudaErrors(cudaStreamSynchronize(stream));
            // std::cout << "cudaStreamSynchronize passed" << std::endl;

            auto duration = duration_cast<microseconds>(stop - start);

            slam_frame->image = image;
            slam_frame->keypoints_count = keypoints_num;

            slam_frame->keypoints_x = keypoints_x;
            slam_frame->keypoints_y = keypoints_y;
            // std::cout << "Keypoints are exported" << std::endl;

            pEvent newEvent = std::make_shared<BaseEvent>();
            newEvent->event_type = EventType::event_gpu_slam_frame;
            newEvent->message = slam_frame;
            _ctx->sendEvent(newEvent);

            slam_frames[slam_frames_id]->image_ready_for_process = false;

            std::lock_guard<std::mutex> lock(m_gpu_mutex);
            deleted_slam_frames.push_back(slam_frames_id);

            std::cout << "Finished work GPU thread " << slam_frames_id
                      << " duration " << duration.count()
                      //   << " keypoints_num " << keypoints_num
                      << std::endl;
        }

        checkCudaErrors(cudaFree(d_rgb_image));
        checkCudaErrors(cudaFree(d_gray_image));
        checkCudaErrors(cudaFree(d_keypoints_exist));
        checkCudaErrors(cudaFree(d_keypoints_response));
        checkCudaErrors(cudaFree(d_keypoints_angle));
        checkCudaErrors(cudaFree(d_descriptors));

        std::cout << "Stopped GPU thread " << slam_frames_id << std::endl;
    }

    void SlamGpuPipeline::handleEvent(pEvent event)
    {

        switch (event->event_type)
        {

        case EventType::event_stop_thread:
        {
            exit_gpu_pipeline = true;
            std::cout << "Stopping GPU threads" << std::endl;
            for (int i = 0; i < _ctx->SlamGpuPipeline_max_streams_length; i++)
            {
                slam_frames[i]->thread_cv.notify_one();
                slam_frames[i]->gpu_thread.join();
            }
            break;
        }

        case EventType::event_realsense_D400_rgbd:
        {
            // upload intinsics/extrinsics to GPU if not uploaded already
            if (!intristics_are_known)
            {
                auto rgbd_frame = std::static_pointer_cast<rgbd_frame_t>(event->message);
                upload_intristics(rgbd_frame);
            }
            else
            {
                auto rgbd_frame = std::static_pointer_cast<rgbd_frame_t>(event->message);
                rgb_curr_frame_id = rgbd_frame->rgb_frame.get_frame_number();
                depth_curr_frame_id = rgbd_frame->depth_frame.get_frame_number();
                // if (frame_counter == _ctx->RealSenseD400_autoexposure_settle_frame - 1)
                // {
                //     auto tt = rgbd_frame->depth_frame.get_sensor();
                // }

                if (deleted_slam_frames.size() > 0 &&
                    frame_counter > _ctx->RealSenseD400_autoexposure_settle_frame &&
                    rgb_curr_frame_id != rgb_prev_frame_id &&
                    depth_curr_frame_id != depth_prev_frame_id)
                {
                    std::lock_guard<std::mutex> lock(m_gpu_mutex);
                    int thread_id = deleted_slam_frames.back();
                    deleted_slam_frames.pop_back();
                    // m_gpu_mutex.unlock();

                    // auto rgbd_frame = std::static_pointer_cast<rgbd_frame_t>(event->message);
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

} // namespace Jetracer