#include "SlamGpuPipeline.h"

#include <memory>
#include <chrono>
#include <iostream>
#include <pthread.h>
// #include "../external/vilib/preprocess/pyramid.h"
// #include "../external/vilib/storage/pyramid_pool.h"
// #include "../external/vilib/feature_detection/fast/fast_common.h"
// #include "../external/vilib/feature_detection/fast/fast_gpu.h"
// #include "../external/vilib/config.h"
// #include "vilib/common/subframe.h"

// #include "../cuda/rscuda_utils.cuh"
#include "../cuda/cuda_RGB_to_Grayscale.cuh"
#include "../cuda/orb.cuh"
#include "../cuda/cuda-align.cuh"
#include "../cuda/pyramid.cuh"
#include "../cuda/fast.cuh"
#include "defines.h"

#include <unistd.h> // for sleep function
#include <chrono>
#include <nvjpeg.h>
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
        checkCudaErrors(cudaMalloc((void **)&_d_rgb_intrinsics, sizeof(rs2_intrinsics)));
        checkCudaErrors(cudaMalloc((void **)&_d_depth_intrinsics, sizeof(rs2_intrinsics)));
        checkCudaErrors(cudaMalloc((void **)&_d_depth_rgb_extrinsics, sizeof(rs2_extrinsics)));

        std::cout << "SlamGpuPipeline is initialized" << std::endl;
    }

    void SlamGpuPipeline::upload_intristics(std::shared_ptr<Jetracer::rgbd_frame_t> rgbd_frame)
    {
        // std::cout << "Uploading intinsics " << std::endl;

        auto rgb_profile = rgbd_frame->rgb_frame.get_profile().as<rs2::video_stream_profile>();
        auto depth_profile = rgbd_frame->depth_frame.get_profile().as<rs2::video_stream_profile>();

        rs2_intrinsics h_rgb_intrinsics = rgb_profile.get_intrinsics();
        rs2_intrinsics h_depth_intrinsics = depth_profile.get_intrinsics();
        rs2_extrinsics h_depth_rgb_extrinsics = depth_profile.get_extrinsics_to(rgb_profile);

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

        depth_scale = rgbd_frame->depth_frame.get_units();
        intristics_are_known = true;
        std::cout << "Uploaded intinsics " << std::endl;
    }

    void SlamGpuPipeline::buildStream(int slam_frames_id)
    {
        // spread threads between cores
        // cpu_set_t cpuset;
        // CPU_ZERO(&cpuset);
        // CPU_SET(slam_frames_id + 1, &cpuset);

        // int rc = pthread_setaffinity_np(slam_frames[slam_frames_id]->gpu_thread.native_handle(),
        //                                 sizeof(cpu_set_t), &cpuset);
        // if (rc != 0)
        // {
        //     std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
        // }

        std::cout
            << "GPU thread " << slam_frames_id << " is started on CPU: "
            << sched_getcpu() << std::endl;

        cudaStream_t stream;
        cudaStream_t align_stream;
        // cudaStream_t io_stream;
        checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        checkCudaErrors(cudaStreamCreateWithFlags(&align_stream, cudaStreamNonBlocking));
        // checkCudaErrors(cudaStreamCreateWithFlags(&io_stream, cudaStreamNonBlocking));

        // allocate memory for image processing
        unsigned char *d_rgb_image;
        unsigned char *d_gray_image;
        unsigned int *d_aligned_out;
        uint16_t *d_depth_in;
        int2 *d_pixel_map;
        // float *d_gray_keypoint_response;
        // bool *d_keypoints_exist;
        // float *d_keypoints_response;
        float *d_keypoints_angle;
        // float2 *d_keypoints_pos;
        unsigned char *d_descriptors;

        int grid_cols = (_ctx->cam_w + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
        int grid_rows = (_ctx->cam_h + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
        int keypoints_num = grid_cols * grid_rows;

        int data_bytes = _ctx->cam_w * _ctx->cam_h * 3;
        std::size_t width_char = sizeof(char) * _ctx->cam_w;
        // std::size_t width_float = sizeof(float) * _ctx->cam_w;
        std::size_t height = _ctx->cam_h;

        std::size_t rgb_pitch;
        std::size_t gray_pitch;
        // std::size_t gray_response_pitch;
        checkCudaErrors(cudaMallocPitch((void **)&d_rgb_image, &rgb_pitch, width_char * 3, height));
        checkCudaErrors(cudaMallocPitch((void **)&d_gray_image, &gray_pitch, width_char, height));
        checkCudaErrors(cudaMalloc((void **)&d_aligned_out, _ctx->cam_w * sizeof(unsigned int) * _ctx->cam_h));
        checkCudaErrors(cudaMalloc((void **)&d_depth_in, _ctx->cam_w * sizeof(uint16_t) * _ctx->cam_h));
        checkCudaErrors(cudaMalloc((void **)&d_pixel_map, _ctx->cam_w * sizeof(int2) * _ctx->cam_h * 2)); // it needs x2 size
        // checkCudaErrors(cudaMalloc((void **)&d_keypoints_exist, keypoints_num * sizeof(bool)));
        // checkCudaErrors(cudaMalloc((void **)&d_keypoints_response, keypoints_num * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_keypoints_angle, keypoints_num * sizeof(float)));
        // checkCudaErrors(cudaMalloc((void **)&d_keypoints_pos, keypoints_num * sizeof(float2)));
        checkCudaErrors(cudaMalloc((void **)&d_descriptors, keypoints_num * 32 * sizeof(unsigned char)));

        unsigned char *d_corner_lut;
        checkCudaErrors(cudaMalloc((void **)&d_corner_lut, 64 * 1024));

        //nvJPEG to encode RGB image to jpeg
        nvjpegHandle_t nv_handle;
        nvjpegEncoderState_t nv_enc_state;
        nvjpegEncoderParams_t nv_enc_params;
        int resize_quality = 90;

        CHECK_NVJPEG(nvjpegCreateSimple(&nv_handle));
        CHECK_NVJPEG(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));
        CHECK_NVJPEG(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));
        CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(nv_enc_params, resize_quality, stream));
        CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_420, stream));
        // CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, 0, NULL));
        nvjpegImage_t nv_image;

        /*
        * Preallocate FeaturePoint struct of arrays
        * Note to future self:
        * we use SoA, because of the efficient bearing vector calculation
        * float x_
        * float y_:                                      | 2x float
        * float score_:                                  | 1x float
        * int level_:                                    | 1x int
        */
        const std::size_t bytes_per_featurepoint = sizeof(float) * 4;
        std::size_t feature_cell_count = grid_cols * grid_rows;
        std::size_t feature_grid_bytes = feature_cell_count * bytes_per_featurepoint;

        float *d_feature_grid;
        checkCudaErrors(cudaMalloc((void **)&d_feature_grid, feature_grid_bytes));
        float2 *d_pos = (float2 *)(d_feature_grid);
        float *d_score = (d_feature_grid + feature_cell_count * 2);
        int *d_level = (int *)(d_feature_grid + feature_cell_count * 3);

        std::shared_ptr<uint16_t[]> keypoints_x(new uint16_t[keypoints_num]);
        std::shared_ptr<uint16_t[]> keypoints_y(new uint16_t[keypoints_num]);

        // Preparing image pyramid
        std::vector<pyramid_t> pyramid;
        int prev_width;
        int prev_height;
        for (int i = 0; i < PYRAMID_LEVELS; i++)
        {
            pyramid_t level;
            if (i != 0)
            {
                level.image_width = prev_width / 2;
                level.image_height = prev_height / 2;
            }
            else
            {
                level.image_width = _ctx->cam_w;
                level.image_height = _ctx->cam_h;
            }
            checkCudaErrors(cudaMallocPitch((void **)&level.image,
                                            &level.image_pitch,
                                            level.image_width * sizeof(char),
                                            level.image_height));
            checkCudaErrors(cudaMallocPitch((void **)&level.response,
                                            &level.response_pitch,
                                            level.image_width * sizeof(float),
                                            level.image_height));
            pyramid.push_back(level);
            prev_width = level.image_width;
            prev_height = level.image_height;
        }

        fast_gpu_calculate_lut(d_corner_lut, FAST_MIN_ARC_LENGTH);

        while (!exit_gpu_pipeline)
        {
            std::unique_lock<std::mutex> lk(slam_frames[slam_frames_id]->thread_mutex);
            while (!slam_frames[slam_frames_id]->image_ready_for_process)
                slam_frames[slam_frames_id]->thread_cv.wait(lk);

            if (!slam_frames[slam_frames_id]->image_ready_for_process)
                continue;

            std::shared_ptr<Jetracer::slam_frame_t> slam_frame = std::make_shared<slam_frame_t>();
            // slam_frame->image = (unsigned char *)malloc(_ctx->cam_h * _ctx->cam_w * sizeof(char));

            std::shared_ptr<float[]> h_feature_grid(new float[feature_cell_count * 4]);
            float2 *h_pos = (float2 *)(h_feature_grid.get());
            float *h_score = reinterpret_cast<float *>(h_feature_grid.get() + feature_cell_count * 2);
            int *h_level = (int *)(h_feature_grid.get() + feature_cell_count * 3);

            std::chrono::_V2::system_clock::time_point stop;
            std::chrono::_V2::system_clock::time_point start = high_resolution_clock::now();

            // --------------- align depth to RGB -----------------
            checkCudaErrors(cudaMemcpyAsync((void *)d_depth_in,
                                            slam_frames[slam_frames_id]->rgbd_frame->depth_frame.get_data(),
                                            _ctx->cam_w * sizeof(uint16_t) * _ctx->cam_h,
                                            cudaMemcpyHostToDevice,
                                            align_stream));
            //                              align_stream));

            // std::cout << "----> align_depth_to_other" << std::endl;
            align_depth_to_other(d_aligned_out,
                                 d_depth_in,
                                 d_pixel_map,
                                 depth_scale,
                                 _ctx->cam_w,
                                 _ctx->cam_h,
                                 _d_depth_intrinsics,
                                 _d_rgb_intrinsics,
                                 _d_depth_rgb_extrinsics,
                                 align_stream);
            // std::cout << "<---- align_depth_to_other" << std::endl;

            // Align depth with Color images and Keypoints compute are going in parallel CUDA streams for better GPU utilization
            //--------------- Keypoints find/compute -------------
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

            gaussian_blur_3x3(pyramid[0].image,
                              pyramid[0].image_pitch,
                              d_gray_image,
                              gray_pitch,
                              _ctx->cam_w,
                              _ctx->cam_h,
                              stream);

            // checkCudaErrors(cudaMemcpy2DAsync((void *)slam_frame->image,
            //                                   _ctx->cam_w,
            //                                   pyramid[0].image,
            //                                   pyramid[0].image_pitch,
            //                                   _ctx->cam_w,
            //                                   _ctx->cam_h,
            //                                   cudaMemcpyDeviceToHost,
            //                                   stream));

            pyramid_create_levels(pyramid, stream);

            detect(pyramid,
                   d_corner_lut,
                   FAST_EPSILON,
                   d_pos,
                   d_score,
                   d_level,
                   stream);

            compute_fast_angle(d_keypoints_angle,
                               d_pos,
                               pyramid[0].image,
                               pyramid[0].image_pitch,
                               _ctx->cam_w,
                               _ctx->cam_h,
                               keypoints_num,
                               stream);

            calc_orb(d_keypoints_angle,
                     d_pos,
                     d_descriptors,
                     pyramid[0].image,
                     pyramid[0].image_pitch,
                     _ctx->cam_w,
                     _ctx->cam_h,
                     keypoints_num,
                     stream);

            checkCudaErrors(cudaMemcpyAsync((void *)h_feature_grid.get(),
                                            d_feature_grid,
                                            feature_grid_bytes,
                                            cudaMemcpyDeviceToHost,
                                            stream));

            // Fill nv_image with image data, let's say 848x480 image in RGB format
            for (int i = 0; i < 3; i++)
            {
                nv_image.channel[i] = pyramid[0].image;
                nv_image.pitch[i] = pyramid[0].image_pitch;
                // checkCudaErrors(cudaMalloc((void **)&(nv_image.channel[i]), image_channel_size));
                // nv_image.pitch[i] = _ctx->cam_w;
                // checkCudaErrors(cudaMemcpy(nv_image.channel[i], casted_image + image_channel_size * i, image_channel_size, cudaMemcpyHostToDevice));
            }

            // Compress image
            CHECK_NVJPEG(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
                                           &nv_image, NVJPEG_INPUT_RGB, _ctx->cam_w, _ctx->cam_h, stream));

            // get compressed stream size
            size_t length;
            // std::cout << "<---- nvjpegEncodeRetrieveBitstream length" << std::endl;
            CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream));
            // get stream itself
            checkCudaErrors(cudaStreamSynchronize(stream));
            slam_frame->image = (unsigned char *)malloc(length * sizeof(char));
            slam_frame->image_length = length;
            // std::cout << "<---- nvjpegEncodeRetrieveBitstream slam_frame->image, length: " << length << std::endl;
            CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, (slam_frame->image), &length, stream));
            checkCudaErrors(cudaStreamSynchronize(stream));
            checkCudaErrors(cudaStreamSynchronize(align_stream));
            stop = high_resolution_clock::now();

            // copy keypoints to slam_frame
#pragma unroll
            slam_frame->keypoints_count = 0;
            for (int i = 0; i < keypoints_num; i++)
            {
                keypoints_x[i] = uint16_t(h_pos[i].x);
                keypoints_y[i] = uint16_t(h_pos[i].y);
            }

            // sleep(2);
            // std::cout << "cudaStreamSynchronize" << std::endl;
            // checkCudaErrors(cudaStreamSynchronize(stream));
            // std::cout << "cudaStreamSynchronize passed" << std::endl;

            auto duration = duration_cast<microseconds>(stop - start);

            // slam_frame->image = image;
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
        checkCudaErrors(cudaFree(d_aligned_out));
        checkCudaErrors(cudaFree(d_depth_in));
        // checkCudaErrors(cudaFree(d_keypoints_exist));
        // checkCudaErrors(cudaFree(d_keypoints_response));
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