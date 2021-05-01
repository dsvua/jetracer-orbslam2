#include "SaveRawData.h"
#include "RealSenseD400.h"

#include <memory>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string.h>
#include <cstring> // for memcpy

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
// #include <nvjpeg.h>

// using namespace std;

namespace Jetracer
{
    // #define CHECK_NVJPEG(call)                                                                                  \
//     {                                                                                                       \
//         nvjpegStatus_t _e = (call);                                                                         \
//         if (_e != NVJPEG_STATUS_SUCCESS)                                                                    \
//         {                                                                                                   \
//             std::cout << "NVJPEG failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
//             exit(1);                                                                                        \
//         }                                                                                                   \
//     }

    // #define CHECK_CUDA(call)                                                                                          \
//     {                                                                                                             \
//         cudaError_t _e = (call);                                                                                  \
//         if (_e != cudaSuccess)                                                                                    \
//         {                                                                                                         \
//             std::cout << "CUDA Runtime failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
//             exit(1);                                                                                              \
//         }                                                                                                         \
//     }

    SaveRawData::SaveRawData(const std::string threadName, context_t *ctx) : EventsThread(threadName), _ctx(ctx)
    {
        auto pushEventCallback = [this](pEvent event) -> bool {
            this->pushEvent(event);
            return true;
        };

        _ctx->subscribeForEvent(EventType::event_realsense_D400_rgbd, threadName, pushEventCallback);
        // _ctx->subscribeForEvent(EventType::event_realsense_D400_rgbd, threadName, pushEventCallback);

        std::cout << "SaveRawData is initialized" << std::endl;
    }

    void SaveRawData::handleEvent(pEvent event)
    {

        switch (event->event_type)
        {

        // TODO: jpeg encoding produce incorrect images, needs to be fixed
        case EventType::event_realsense_D400_rgb:
        {
            // initialize nvjpeg structures
            // std::shared_ptr<rgbd_frame_t> rgb_frame = std::static_pointer_cast<rgbd_frame_t>(event->message);

            //nvJPEG to encode RGB image to jpeg
            // nvjpegHandle_t nv_handle;
            // nvjpegEncoderState_t nv_enc_state;
            // nvjpegEncoderParams_t nv_enc_params;
            // cudaStream_t stream;
            // int resize_quality = 90;

            // CHECK_NVJPEG(nvjpegCreateSimple(&nv_handle));
            // CHECK_NVJPEG(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));
            // CHECK_NVJPEG(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));
            // CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(nv_enc_params, resize_quality, stream));
            // CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_444, stream));

            // CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, 0, NULL));

            // nvjpegImage_t nv_image;
            // const unsigned char *casted_image = static_cast<const unsigned char *>(rgb_frame->depth_frame.get_data());
            // int image_channel_size = _ctx->cam_w * _ctx->cam_h;
            // Fill nv_image with image data, let's say 848x480 image in RGB format
            // for (int i = 0; i < 3; i++)
            // {
            //     CHECK_CUDA(cudaMalloc((void **)&(nv_image.channel[i]), image_channel_size));
            //     nv_image.pitch[i] = _ctx->cam_w;
            //     CHECK_CUDA(cudaMemcpy(nv_image.channel[i], casted_image + image_channel_size * i, image_channel_size, cudaMemcpyHostToDevice));
            // }

            // Compress image
            // CHECK_NVJPEG(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
            //                                &nv_image, NVJPEG_INPUT_RGB, _ctx->cam_w, _ctx->cam_h, stream));

            // get compressed stream size
            // size_t length;
            // CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream));
            // get stream itself
            // CHECK_CUDA(cudaStreamSynchronize(stream));
            // std::vector<unsigned char> jpeg(length);
            // CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, jpeg.data(), &length, 0));

            // std::string filename;

            // filename = _ctx->images_path + "rgb_" + std::to_string(rgb_frame->rgb_frame.get_timestamp()) + ".jpg";
            // std::cout << "Saving file " << filename << " size " << std::to_string(_ctx->cam_w * _ctx->cam_h * 3) << std::endl;

            // write stream to file
            // CHECK_CUDA(cudaStreamSynchronize(stream));
            // std::ofstream jpeg_file(filename, std::ios::out | std::ios::binary);
            // jpeg_file.write(reinterpret_cast<const char *>(jpeg.data()), length);
            // jpeg_file.close();

            break;
        }

        case EventType::event_realsense_D400_rgbd:
        {
            // std::shared_ptr<rs2::frame> temp_frame = std::static_pointer_cast<rs2::frame>(event->message);
            // rs2::frameset fs = temp_frame->as<rs2::frameset>();
            std::shared_ptr<rgbd_frame_t> rgbd_frame = std::static_pointer_cast<rgbd_frame_t>(event->message);

            std::string filename;

            filename = _ctx->images_path + "color_" + std::to_string(rgbd_frame->timestamp) + ".bin";
            // std::cout << "Saving file " << filename << " size " << std::to_string(rgbd_frame->image_size) << std::endl;
            std::ofstream image_file(filename, std::ofstream::binary);
            // const char *image_buffer = static_cast<const char *>(rgbd_frame->rgb_image);
            char *image_buffer = (char *)(rgbd_frame->rgb_image);
            image_file.write(image_buffer, _ctx->cam_w * _ctx->cam_h * 3);
            image_file.close();

            filename = _ctx->images_path + "depth_" + std::to_string(rgbd_frame->timestamp) + ".bin";
            // std::cout << "Saving file " << filename << " size " << std::to_string(rgbd_frame->depth_size) << std::endl;
            std::ofstream depth_file(filename, std::ofstream::binary);
            // const char *depth_buffer = static_cast<const char *>(rgbd_frame->depth_image);
            char *depth_buffer = (char *)(rgbd_frame->depth_image);
            depth_file.write(depth_buffer, _ctx->cam_w * _ctx->cam_h * 2);
            depth_file.close();

            break;
        }

        case EventType::event_stop_thread:
        {
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