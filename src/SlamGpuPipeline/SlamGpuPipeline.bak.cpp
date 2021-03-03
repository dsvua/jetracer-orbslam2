#include "SlamGpuPipeline.h"

#include <memory>
#include <chrono>
#include <iostream>
#include <pthread.h>
#include <opencv2/cudaimgproc.hpp>
#include "../external/vilib/preprocess/pyramid.h"
#include "../external/vilib/storage/pyramid_pool.h"
#include "../external/vilib/feature_detection/fast/fast_common.h"
#include "../external/vilib/feature_detection/fast/fast_gpu.h"
#include "../external/vilib/config.h"
// #include "vilib/common/subframe.h"

#include "../cuda/rscuda_utils.cuh"
#include "../cuda/cuda_RGB_to_Grayscale.cuh"

#include <chrono>
using namespace std::chrono;

// using namespace std;

namespace Jetracer
{
    // template <typename T>
    // std::shared_ptr<T> make_device_copy(T obj)
    // {
    //     T *d_data;
    //     auto res = cudaMalloc(&d_data, sizeof(T));
    //     if (res != cudaSuccess)
    //         throw std::runtime_error("cudaMalloc failed status: " + res);
    //     cudaMemcpy(d_data, &obj, sizeof(T), cudaMemcpyHostToDevice);
    //     return std::shared_ptr<T>(d_data, [](T *data) { cudaFree(data); });
    // }

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

        // detector = cv::cuda::ORB::create(_ctx->SlamGpuPipeline_max_keypoints_to_search, 1.2f, 8, 31, 0, 2,
        //                                  cv::ORB::HARRIS_SCORE, 31, fastThresh);
        // detector = cv::cuda::ORB::create(512); // CUDA warp is 32 and 512/32=16
        // matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

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
        // cudaStream_t io_stream;
        checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        // checkCudaErrors(cudaStreamCreateWithFlags(&io_stream, cudaStreamNonBlocking));
        // cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
        // cv::cuda::Stream cv_io_stream = cv::cuda::StreamAccessor::wrapStream(io_stream);

        // cv::cuda::GpuMat d_rgb_image;
        // cv::cuda::GpuMat d_image;
        // cv::cuda::GpuMat d_depth;
        // cv::cuda::GpuMat d_aligned_depth;
        // cv::cuda::GpuMat d_keypoints;
        // cv::cuda::GpuMat d_descriptors;
        // cv::cuda::GpuMat d_matches;

        // cv::cuda::GpuMat d_keypoints_tmp;
        // cv::cuda::GpuMat d_descriptors_tmp;

        // cv::cuda::GpuMat rad_i;
        // cv::cuda::GpuMat rad_i_sorted;

        // cv::Mat image;
        // cv::Mat h_keypoints;
        // cv::Mat h_descriptors;

        std::shared_ptr<rgbd_frame_t> rgbd_frame;
        std::shared_ptr<int2> d_pixel_map;

        bool image_ready_for_process = false;
        std::thread gpu_thread;
        std::mutex thread_mutex;
        std::condition_variable thread_cv;

        //-------------------------------------------------------------
        std::shared_ptr<vilib::DetectorBaseGPU> detector_gpu_;
        detector_gpu_.reset(new vilib::FASTGPU(_ctx->cam_w,
                                               _ctx->cam_h,
                                               32,
                                               32,
                                               0,
                                               1,
                                               0,
                                               0,
                                               10.0f,
                                               10,
                                               vilib::SUM_OF_ABS_DIFF_ON_ARC));
        detector_gpu_->setStream(stream);

        // // Initialize the pyramid pool
        // // Vector holding the image pyramid either in host or GPU memory
        std::vector<std::shared_ptr<vilib::Subframe>> pyramid_;

        vilib::PyramidPool::init(1,
                                 _ctx->cam_w,
                                 _ctx->cam_h,
                                 1, // grayscale
                                 3,
                                 vilib::IMAGE_PYRAMID_MEMORY_TYPE);

        vilib::PyramidPool::get(IMAGE_PYRAMID_PREALLOCATION_ITEM_NUM,
                                _ctx->cam_w,
                                _ctx->cam_h,
                                1,
                                1,
                                vilib::IMAGE_PYRAMID_MEMORY_TYPE,
                                pyramid_);
        // -------------------------------------------------------------

        // cv::Ptr<cv::cuda::ORB> detector = cv::cuda::ORB::create(_ctx->SlamGpuPipeline_max_keypoints_to_search,
        //                                                         1.2f,
        //                                                         8,
        //                                                         31,
        //                                                         0,
        //                                                         2,
        //                                                         cv::ORB::HARRIS_SCORE,
        //                                                         31,
        //                                                         fastThresh);

        std::cout
            << "GPU thread " << slam_frames_id << " is started on CPU: "
            << sched_getcpu() << std::endl;

        while (!exit_gpu_pipeline)
        {
            std::unique_lock<std::mutex> lk(thread_mutex);
            while (!slam_frames[slam_frames_id]->image_ready_for_process)
                slam_frames[slam_frames_id]->thread_cv.wait(lk);

            if (!slam_frames[slam_frames_id]->image_ready_for_process)
                continue;

            cv::Size sz(_ctx->cam_w, _ctx->cam_h);

            // std::cout << "Working on GPU pipeline " << slam_frames_id << std::endl;
            auto slam_frame = std::make_shared<slam_frame_t>();

            // void *char_data = const_cast<void *>(rgbd_frame->lefr_ir);
            // slam_frames[slam_frames_id]->image = cv::Mat(sz, CV_8UC1, char_data);
            // std::cout << "---Mark---cvtColor, thread " << slam_frames_id << std::endl;

            // cv::cuda::cvtColor(slam_frames[slam_frames_id]->rgbd_frame->d_rgb_image,
            //                    d_image,
            //                    cv::COLOR_RGB2GRAY,
            //                    1,
            //                    cv_stream);

            // d_image.create(slam_frames[slam_frames_id]->rgbd_frame->d_rgb_image.size(), CV_8U);
            int data_bytes = slam_frames[slam_frames_id]->rgbd_frame->rgb_frame.get_data_size();
            // std::cout << "---Mark---data_bytes " << data_bytes << std::endl;

            // unsigned char *d_gray_image;
            // checkCudaErrors(cudaMalloc(&d_gray_image, sizeof(unsigned char) * data_bytes / 3));

            unsigned char *d_rgb_image;
            checkCudaErrors(cudaMalloc(&d_rgb_image, sizeof(unsigned char) * data_bytes));
            checkCudaErrors(cudaMemcpyAsync(d_rgb_image,
                                            slam_frames[slam_frames_id]->rgbd_frame->rgb_frame.get_data(),
                                            sizeof(unsigned char) * data_bytes,
                                            cudaMemcpyHostToDevice,
                                            stream));

            rgb_to_grayscale(pyramid_[0]->data_,
                             d_rgb_image,
                             _ctx->cam_w,
                             _ctx->cam_h,
                             pyramid_[0]->pitch_,
                             stream);

            // std::cout << "---Mark---AlignDepthToRGB, thread " << slam_frames_id << std::endl;
            // AlignDepthToRGB(cv_stream,
            //                 d_aligned_depth,
            //                 slam_frames[slam_frames_id]->rgbd_frame);

            // std::cout << "---Mark---d_image.download, thread " << slam_frames_id << std::endl;
            // d_image.download(image,
            //                  cv_stream);

            cv::Mat image(_ctx->cam_h, _ctx->cam_w, CV_8UC1);

            // Do the copying
            // void *dst = (void *)image.data;
            // std::size_t dpitch = image.step;
            checkCudaErrors(cudaMemcpy2DAsync((void *)image.data, image.step,
                                              pyramid_[0]->data_, pyramid_[0]->pitch_,
                                              _ctx->cam_w, _ctx->cam_h,
                                              cudaMemcpyDeviceToHost, stream));

            // checkCudaErrors(cudaStreamSynchronize(stream));
            // std::cout << "---Mark---image.type() " << image.type() << std::endl;

            //-------------------------------------------------------------
            //Do the copying
            // const void *src = d_image.ptr<const void *>(0);
            // void *dst = (void *)pyramid_[0]->data_;
            // std::size_t dpitch = pyramid_[0]->pitch_;
            // std::size_t spitch = d_image.step;
            // std::size_t width = d_image.cols * d_image.elemSize();
            // std::size_t height = d_image.rows;
            // checkCudaErrors(cudaMemcpy2DAsync(dst,
            //                                   dpitch,
            //                                   src,
            //                                   spitch,
            //                                   width,
            //                                   height,
            //                                   cudaMemcpyDeviceToDevice,
            //                                   stream));

            vilib::pyramid_create_gpu(pyramid_, stream);

            // // features detection
            detector_gpu_->reset();
            detector_gpu_->detect(pyramid_);
            auto &keypoints = detector_gpu_->getPoints();
            std::cout << "Points found: " << keypoints.size() << std::endl;

            std::shared_ptr<double[]> keypoints_x(new double[keypoints.size()]);
            std::shared_ptr<double[]> keypoints_y(new double[keypoints.size()]);

            slam_frame->keypoints_count = keypoints.size();

            for (int i = 0; i < keypoints.size(); i++)
            {
                keypoints_x[i] = keypoints[i].x_;
                keypoints_y[i] = keypoints[i].y_;
            }
            slam_frame->keypoints_x = keypoints_x;
            slam_frame->keypoints_y = keypoints_y;

            //-------------------------------------------------------------

            // checkCudaErrors(cudaStreamSynchronize(slam_frames[slam_frames_id]->stream));
            // std::cout << "---Mark---detectAndComputeAsync, thread " << slam_frames_id << std::endl;
            // detector->detectAndComputeAsync(d_image,
            //                                 cv::noArray(),
            //                                 d_keypoints,
            //                                 d_descriptors,
            //                                 false,
            //                                 cv_stream);

            // if (include_anms)
            // {
            //     AdaptiveNonMaximalSuppression(stream,
            //                                   d_keypoints,
            //                                   d_descriptors,
            //                                   rad_i,
            //                                   rad_i_sorted,
            //                                   d_keypoints_tmp,
            //                                   d_descriptors_tmp);
            // }

            // end of pipeline, downloading data and sending message
            // to other subsystems

            // std::cout << "cudaStreamSynchronize " << slam_frames_id << std::endl;
            // checkCudaErrors(cudaStreamSynchronize(slam_frames[slam_frames_id]->stream));

            slam_frame->rgbd_frame = slam_frames[slam_frames_id]->rgbd_frame;
            // slam_frame->d_descriptors = d_descriptors;

            // if (include_anms)
            // {
            //     d_descriptors_tmp.download(slam_frame->descriptors,
            //                                cv_stream);
            //     d_keypoints_tmp.download(h_keypoints,
            //                              cv_stream);

            //     checkCudaErrors(cudaStreamSynchronize(stream));
            //     detector->convert(h_keypoints,
            //                       slam_frame->keypoints);
            // }
            // else
            // {
            //     d_descriptors.download(slam_frame->descriptors,
            //                            cv_stream);
            //     d_keypoints.download(h_keypoints,
            //                          cv_stream);

            //     checkCudaErrors(cudaStreamSynchronize(stream));
            //     detector->convert(h_keypoints,
            //                       slam_frame->keypoints);
            // }

            slam_frame->image = image;

            pEvent newEvent = std::make_shared<BaseEvent>();
            newEvent->event_type = EventType::event_gpu_slam_frame;
            newEvent->message = slam_frame;

            _ctx->sendEvent(newEvent);

            slam_frames[slam_frames_id]->image_ready_for_process = false;

            std::lock_guard<std::mutex> lock(m_gpu_mutex);
            deleted_slam_frames.push_back(slam_frames_id);
            checkCudaErrors(cudaFree(d_rgb_image));
            std::cout << "Finished work GPU thread " << slam_frames_id << std::endl;
            // m_gpu_mutex.unlock();
        }

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