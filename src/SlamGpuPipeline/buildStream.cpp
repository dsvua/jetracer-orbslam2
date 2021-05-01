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
#include <numeric>

using namespace std::chrono;
using namespace Eigen;
using namespace std;

namespace Jetracer
{
    Eigen::Matrix4d best_fit_transform(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
    {
        /*
        Notice:
        1/ JacobiSVD return U,S,V, S as a vector, "use U*S*Vt" to get original Matrix;
        2/ matrix type 'MatrixXd' or 'MatrixXf' matters.
        */
        Eigen::Matrix4d T = Eigen::MatrixXd::Identity(4, 4);
        Eigen::Vector3d centroid_A(0, 0, 0);
        Eigen::Vector3d centroid_B(0, 0, 0);
        Eigen::MatrixXd AA = A;
        Eigen::MatrixXd BB = B;
        int row = A.rows();

        for (int i = 0; i < row; i++)
        {
            centroid_A += A.block<1, 3>(i, 0).transpose();
            centroid_B += B.block<1, 3>(i, 0).transpose();
        }
        centroid_A /= row;
        centroid_B /= row;
        for (int i = 0; i < row; i++)
        {
            AA.block<1, 3>(i, 0) = A.block<1, 3>(i, 0) - centroid_A.transpose();
            BB.block<1, 3>(i, 0) = B.block<1, 3>(i, 0) - centroid_B.transpose();
        }

        Eigen::MatrixXd H = AA.transpose() * BB;
        Eigen::MatrixXd U;
        Eigen::VectorXd S;
        Eigen::MatrixXd V;
        Eigen::MatrixXd Vt;
        Eigen::Matrix3d R;
        Eigen::Vector3d t;

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        U = svd.matrixU();
        S = svd.singularValues();
        V = svd.matrixV();
        Vt = V.transpose();

        R = Vt.transpose() * U.transpose();

        if (R.determinant() < 0)
        {
            std::cout << "R.determinant() < 0" << std::endl;
            Vt.block<1, 3>(2, 0) *= -1;
            R = Vt.transpose() * U.transpose();
        }

        t = centroid_B - R * centroid_A;

        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = t;
        return T;
    }

    float dist(const Eigen::Vector3d &pta, const Eigen::Vector3d &ptb)
    {
        return sqrt((pta[0] - ptb[0]) * (pta[0] - ptb[0]) + (pta[1] - ptb[1]) * (pta[1] - ptb[1]) + (pta[2] - ptb[2]) * (pta[2] - ptb[2]));
    }

    typedef struct
    {
        std::vector<float> distances;
        std::vector<int> indices;
    } NEIGHBOR;

    NEIGHBOR nearest_neighbot(const Eigen::MatrixXd &src, const Eigen::MatrixXd &dst)
    {
        int row_src = src.rows();
        int row_dst = dst.rows();
        Eigen::Vector3d vec_src;
        Eigen::Vector3d vec_dst;
        NEIGHBOR neigh;
        float min = 100;
        int index = 0;
        float dist_temp = 0;

        for (int ii = 0; ii < row_src; ii++)
        {
            vec_src = src.block<1, 3>(ii, 0).transpose();
            min = 100;
            index = 0;
            dist_temp = 0;
            for (int jj = 0; jj < row_dst; jj++)
            {
                vec_dst = dst.block<1, 3>(jj, 0).transpose();
                dist_temp = dist(vec_src, vec_dst);
                if (dist_temp < min)
                {
                    min = dist_temp;
                    index = jj;
                }
            }
            // cout << min << " " << index << endl;
            // neigh.distances[ii] = min;
            // neigh.indices[ii] = index;
            neigh.distances.push_back(min);
            neigh.indices.push_back(index);
        }

        return neigh;
    }

    Eigen::Matrix4d icp(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, int max_iterations, int tolerance)
    {
        int row = A.rows();
        Eigen::MatrixXd src = Eigen::MatrixXd::Ones(3 + 1, row);
        Eigen::MatrixXd src3d = Eigen::MatrixXd::Ones(3, row);
        Eigen::MatrixXd dst = Eigen::MatrixXd::Ones(3 + 1, row);
        NEIGHBOR neighbor;
        Eigen::Matrix4d T;
        Eigen::MatrixXd dst_chorder = Eigen::MatrixXd::Ones(3, row);
        // ICP_OUT result;
        int iter = 0;

        for (int i = 0; i < row; i++)
        {
            src.block<3, 1>(0, i) = A.block<1, 3>(i, 0).transpose();
            src3d.block<3, 1>(0, i) = A.block<1, 3>(i, 0).transpose();
            dst.block<3, 1>(0, i) = B.block<1, 3>(i, 0).transpose();
        }

        double prev_error = 0;
        double mean_error = 0;
        for (int i = 0; i < max_iterations; i++)
        {
            neighbor = nearest_neighbot(src3d.transpose(), B);

            for (int j = 0; j < row; j++)
            {
                dst_chorder.block<3, 1>(0, j) = dst.block<3, 1>(0, neighbor.indices[j]);
            }

            T = best_fit_transform(src3d.transpose(), dst_chorder.transpose());

            src = T * src;
            for (int j = 0; j < row; j++)
            {
                src3d.block<3, 1>(0, j) = src.block<3, 1>(0, j);
            }

            mean_error = std::accumulate(neighbor.distances.begin(), neighbor.distances.end(), 0.0) / neighbor.distances.size();
            if (abs(prev_error - mean_error) < tolerance)
            {
                break;
            }
            prev_error = mean_error;
            iter = i + 2;
        }

        T = best_fit_transform(A, src3d.transpose());
        // result.trans = T;
        // result.distances = neighbor.distances;
        // result.iter = iter;

        // return result;
        return T;
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
        cudaStream_t nvjpeg_stream;
        checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        checkCudaErrors(cudaStreamCreateWithFlags(&align_stream, cudaStreamNonBlocking));
        checkCudaErrors(cudaStreamCreateWithFlags(&nvjpeg_stream, cudaStreamNonBlocking));

        // allocate memory for image processing
        unsigned char *d_rgb_image;
        unsigned char *d_gray_image;
        unsigned char *d_nvjpeg_rgb_image;
        unsigned int *d_aligned_out;
        uint16_t *d_depth_in;
        int2 *d_pixel_map;
        // float *d_gray_keypoint_response;
        // bool *d_keypoints_exist;
        // float *d_keypoints_response;
        float *d_keypoints_angle;
        // float2 *d_keypoints_pos;
        unsigned char *d_descriptors_tmp;
        uint32_t *d_descriptors; // convert 32xchars into 1 x uint32_t
        int *d_valid_keypoints_num;
        int *d_keypoints_num_matched;
        int h_valid_keypoints_num = 0;

        int grid_cols = (_ctx->cam_w + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
        int grid_rows = (_ctx->cam_h + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE;
        std::size_t keypoints_num = grid_cols * grid_rows;

        int data_bytes = _ctx->cam_w * _ctx->cam_h * 3;
        size_t length;
        std::size_t width_char = sizeof(char) * _ctx->cam_w;
        // std::size_t width_float = sizeof(float) * _ctx->cam_w;
        std::size_t height = _ctx->cam_h;

        std::size_t rgb_pitch;
        std::size_t gray_pitch;
        std::size_t nvjpeg_rgb_pitch;
        // std::size_t gray_response_pitch;
        checkCudaErrors(cudaMallocPitch((void **)&d_rgb_image, &rgb_pitch, width_char * 3, height));
        checkCudaErrors(cudaMallocPitch((void **)&d_gray_image, &gray_pitch, width_char, height));
        checkCudaErrors(cudaMallocPitch((void **)&d_nvjpeg_rgb_image, &nvjpeg_rgb_pitch, width_char, height * 3));
        checkCudaErrors(cudaMalloc((void **)&d_aligned_out, _ctx->cam_w * sizeof(unsigned int) * _ctx->cam_h));
        checkCudaErrors(cudaMalloc((void **)&d_depth_in, _ctx->cam_w * sizeof(uint16_t) * _ctx->cam_h));
        checkCudaErrors(cudaMalloc((void **)&d_pixel_map, _ctx->cam_w * sizeof(int2) * _ctx->cam_h * 2)); // it needs x2 size
        // checkCudaErrors(cudaMalloc((void **)&d_keypoints_exist, keypoints_num * sizeof(bool)));
        // checkCudaErrors(cudaMalloc((void **)&d_keypoints_response, keypoints_num * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_keypoints_angle, keypoints_num * sizeof(float)));
        // checkCudaErrors(cudaMalloc((void **)&d_keypoints_pos, keypoints_num * sizeof(float2)));
        checkCudaErrors(cudaMalloc((void **)&d_descriptors_tmp, keypoints_num * 32 * sizeof(unsigned char)));
        checkCudaErrors(cudaMalloc((void **)&d_descriptors, keypoints_num * sizeof(uint32_t)));
        checkCudaErrors(cudaMalloc((void **)&d_valid_keypoints_num, sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_keypoints_num_matched, sizeof(int)));

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
        * float3 point                                   | 3x float
        */
        const std::size_t bytes_per_featurepoint = sizeof(float) * 4;
        std::size_t feature_grid_bytes = keypoints_num * bytes_per_featurepoint;

        float *d_feature_grid;
        checkCudaErrors(cudaMalloc((void **)&d_feature_grid, feature_grid_bytes));
        float2 *d_pos = (float2 *)(d_feature_grid);
        float *d_score = (d_feature_grid + keypoints_num * 2);
        int *d_level = (int *)(d_feature_grid + keypoints_num * 3);
        // double *d_points;
        // double *d_points_prev;
        double *d_matched_points;
        double *d_matched_points_prev;
        int *d_matched_points_num; // should make sure there 3 or more points matched
        // checkCudaErrors(cudaMalloc((void **)&d_points, keypoints_num * 3 * sizeof(double)));
        // checkCudaErrors(cudaMalloc((void **)&d_points_prev, keypoints_num * 3 * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&d_matched_points, keypoints_num * 3 * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&d_matched_points_prev, keypoints_num * 3 * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&d_matched_points_num, sizeof(int)));

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

        int h_valid_keypoints_num_previous = 500;
        Eigen::Matrix4d T_w2c;
        std::shared_ptr<Jetracer::slam_frame_t> previous_frame1 = nullptr;
        std::shared_ptr<Jetracer::slam_frame_t> previous_frame2 = nullptr;

        while (!slam_frames[slam_frames_id]->exit_gpu_pipeline)
        {
            std::unique_lock<std::mutex> lk(slam_frames[slam_frames_id]->thread_mutex);
            while (!slam_frames[slam_frames_id]->image_ready_for_process && !slam_frames[slam_frames_id]->exit_gpu_pipeline)
                slam_frames[slam_frames_id]->thread_cv.wait(lk);

            if (!slam_frames[slam_frames_id]->image_ready_for_process)
                continue;

            std::shared_ptr<Jetracer::slam_frame_t> slam_frame = std::make_shared<slam_frame_t>();
            slam_frame->theta = slam_frames[slam_frames_id]->rgbd_frame->theta;
            // slam_frame->image = (unsigned char *)malloc(_ctx->cam_h * _ctx->cam_w * sizeof(char));

            int h_keypoints_num_matched = 0;

            std::shared_ptr<float[]> h_feature_grid(new float[keypoints_num * 4]);
            float2 *h_pos = (float2 *)(h_feature_grid.get());
            float *h_score = reinterpret_cast<float *>(h_feature_grid.get() + keypoints_num * 2);
            int *h_level = (int *)(h_feature_grid.get() + keypoints_num * 3);
            std::shared_ptr<double[]> h_points(new double[keypoints_num * 3]);
            std::shared_ptr<uint32_t[]> h_descriptors(new uint32_t[keypoints_num]);
            std::shared_ptr<double3[]> h_current_matched_points(new double3[keypoints_num]);
            std::shared_ptr<double3[]> h_previous_matched_points(new double3[keypoints_num]);
            checkCudaErrors(cudaMalloc((void **)&slam_frame->d_points, keypoints_num * 3 * sizeof(double)));
            checkCudaErrors(cudaMalloc((void **)&slam_frame->d_descriptors, keypoints_num * sizeof(uint32_t)));
            checkCudaErrors(cudaMalloc((void **)&slam_frame->d_pos, keypoints_num * sizeof(float2)));

            std::chrono::_V2::system_clock::time_point stop;
            std::chrono::_V2::system_clock::time_point start = high_resolution_clock::now();

            // --------------- align depth to RGB -----------------
            checkCudaErrors(cudaMemcpyAsync((void *)d_depth_in,
                                            slam_frames[slam_frames_id]->rgbd_frame->depth_image,
                                            _ctx->cam_w * sizeof(uint16_t) * _ctx->cam_h,
                                            cudaMemcpyHostToDevice,
                                            align_stream));
            //                              align_stream));

            // Align depth with Color images and Keypoints compute are going in parallel CUDA streams
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

            // Align depth with Color images and Keypoints compute are going in parallel CUDA streams
            // for better GPU utilization
            //--------------- Keypoints find/compute -------------
            checkCudaErrors(cudaMemcpy2DAsync((void *)d_rgb_image,
                                              rgb_pitch,
                                              slam_frames[slam_frames_id]->rgbd_frame->rgb_image,
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
                     d_descriptors_tmp,
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

            keypoint_pixel_to_point(d_aligned_out,
                                    _d_rgb_intrinsics,
                                    _ctx->cam_w,
                                    _ctx->cam_h,
                                    slam_frame->d_pos,
                                    d_pos,
                                    d_score,
                                    slam_frame->d_points,
                                    slam_frame->d_descriptors,
                                    d_descriptors,
                                    keypoints_num,
                                    &h_valid_keypoints_num,
                                    d_valid_keypoints_num,
                                    stream);

            checkCudaErrors(cudaMemcpyAsync((void *)h_points.get(),
                                            slam_frame->d_points,
                                            h_valid_keypoints_num * 3 * sizeof(double),
                                            cudaMemcpyDeviceToHost,
                                            stream));

            checkCudaErrors(cudaStreamSynchronize(align_stream));

            //------------- compressing image for sending over network ------------
            // Fill nv_image with image data, let's say 848x480 image in RGB format
            for (int i = 0; i < 3; i++)
            {
                unsigned char *d_channel_begin = d_nvjpeg_rgb_image + nvjpeg_rgb_pitch * _ctx->cam_h * i;
                checkCudaErrors(cudaMemcpy2DAsync((void *)d_channel_begin,
                                                  nvjpeg_rgb_pitch,
                                                  d_gray_image,
                                                  gray_pitch,
                                                  _ctx->cam_w,
                                                  _ctx->cam_h,
                                                  cudaMemcpyDeviceToDevice,
                                                  nvjpeg_stream));

                nv_image.channel[i] = d_channel_begin;
                nv_image.pitch[i] = nvjpeg_rgb_pitch;
            };

            // overlay keypoints on grayscale or color image
            checkCudaErrors(cudaStreamSynchronize(stream));
            overlay_keypoints(nv_image.channel[1],
                              nv_image.pitch[1],
                              slam_frame->d_pos,
                              d_aligned_out,
                              h_valid_keypoints_num,
                              _d_rgb_intrinsics,
                              nvjpeg_stream);

            // Compress image
            CHECK_NVJPEG(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
                                           &nv_image, NVJPEG_INPUT_RGB, _ctx->cam_w, _ctx->cam_h, nvjpeg_stream));

            if (previous_frame1)
            {
                Eigen::Matrix4d T_w2c_prev_curr = Eigen::Matrix4d::Identity();
                Eigen::Matrix4d T_c2w_prev_curr = Eigen::Matrix4d::Identity();

                if (previous_frame2)
                {

                    // std::cout << "previous_frame2->T_w2c" << std::endl
                    //           << previous_frame2->T_w2c << std::endl;
                    // std::cout << "previous_frame1->T_w2c" << std::endl
                    //           << previous_frame1->T_w2c << std::endl;

                    T_w2c_prev_curr = previous_frame1->T_w2c * previous_frame2->T_w2c.inverse() * previous_frame1->T_w2c;
                    // Eigen::Quaterniond q1(previous_frame1->T_w2c.rotation());
                }

                T_w2c_prev_curr.setIdentity();

                // std::cout << "T_w2c_prev_curr" << std::endl
                //           << T_w2c_prev_curr << std::endl;

                match_keypoints(slam_frame,
                                previous_frame1,
                                2,
                                4,
                                T_w2c_prev_curr,
                                d_valid_keypoints_num,
                                d_keypoints_num_matched,
                                &h_keypoints_num_matched,
                                h_current_matched_points.get(),
                                h_previous_matched_points.get(),
                                _d_rgb_intrinsics,
                                stream);

                if (h_keypoints_num_matched > 2)
                {
                    Eigen::Map<Eigen::MatrixXd> current_points((double *)h_current_matched_points.get(), 3, h_keypoints_num_matched);
                    Eigen::Map<Eigen::MatrixXd> previous_points((double *)h_previous_matched_points.get(), 3, h_keypoints_num_matched);

                    // Eigen::MatrixXd A = previous_points.transpose();
                    // Eigen::MatrixXd B = current_points.transpose();

                    // std::cout << "Matching points" << std::endl;
                    // for (int i = 0; i < h_keypoints_num_matched; i++)
                    // {
                    //     std::cout << A.block<1, 3>(i, 0) << "\t\t" << B.block<1, 3>(i, 0) << std::endl;
                    // }

                    // T_c2w_prev_curr = best_fit_transform(previous_points.transpose(), current_points.transpose());
                }
                else
                {
                    T_w2c_prev_curr.setIdentity();
                    // slam_frame->T_c2w = Eigen::Matrix4d::Identity();
                    // slam_frame->T_w2c = Eigen::Matrix4d::Identity();
                }

                // slam_frame->T_c2w = T_c2w_prev_curr * previous_frame1->T_c2w;
                // slam_frame->T_w2c = slam_frame->T_c2w.inverse();
                slam_frame->T_c2w = Eigen::Matrix4d::Identity();
                slam_frame->T_w2c = Eigen::Matrix4d::Identity();

                // getting rotation angles
                Eigen::Matrix3d R = T_c2w_prev_curr.block<3, 3>(0, 0);
                Vector3d eu_angles = R.eulerAngles(0, 1, 2);
                Vector3d angles(floor(eu_angles(0) * 180 / CUDART_PI_D),
                                floor(eu_angles(1) * 180 / CUDART_PI_D),
                                floor(eu_angles(2) * 180 / CUDART_PI_D));

                std::cout << std::endl
                          << "eu_angles" << std::endl
                          << eu_angles << std::endl
                          << "angles" << std::endl
                          << angles << std::endl
                          << "T_c2w_prev_curr" << std::endl
                          << T_c2w_prev_curr << std::endl
                          << "slam_frame->T_c2w" << std::endl
                          << slam_frame->T_c2w << std::endl;
            }
            else
            {
                slam_frame->T_c2w = Eigen::Matrix4d::Identity();
                slam_frame->T_w2c = Eigen::Matrix4d::Identity();
                h_keypoints_num_matched = keypoints_num;
            }
            // -------------------------------------------------
            // TODO: convert cudaStreamSynchronize to use events
            // -------------------------------------------------

            // download compressed image to host
            checkCudaErrors(cudaStreamSynchronize(nvjpeg_stream));
            // get compressed stream size
            CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, nvjpeg_stream));
            // get stream itself
            slam_frame->image = (unsigned char *)malloc(length * sizeof(char));
            slam_frame->image_length = length;
            CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, (slam_frame->image), &length, nvjpeg_stream));
            checkCudaErrors(cudaStreamSynchronize(nvjpeg_stream));
            checkCudaErrors(cudaStreamSynchronize(stream));

            stop = high_resolution_clock::now();
            auto rotation_timer = high_resolution_clock::now();

            // sleep(2);
            // std::cout << "cudaStreamSynchronize" << std::endl;
            // checkCudaErrors(cudaStreamSynchronize(stream));
            // std::cout << "cudaStreamSynchronize passed" << std::endl;

            auto duration = duration_cast<microseconds>(stop - start);
            auto duration_rotation = duration_cast<microseconds>(rotation_timer - stop);

            // slam_frame->image = image;
            slam_frame->keypoints_count = keypoints_num;
            slam_frame->h_matched_keypoints_num = h_keypoints_num_matched;
            slam_frame->h_points = h_points;
            slam_frame->h_valid_keypoints_num = h_valid_keypoints_num;

            pEvent newEvent = std::make_shared<BaseEvent>();
            newEvent->event_type = EventType::event_gpu_slam_frame;
            newEvent->message = slam_frame;
            if (previous_frame2)
                _ctx->sendEvent(newEvent);

            slam_frames[slam_frames_id]->image_ready_for_process = false;

            std::lock_guard<std::mutex> lock(m_gpu_mutex);
            deleted_slam_frames.push_back(slam_frames_id);

            if (previous_frame1)
                previous_frame2 = previous_frame1;
            previous_frame1 = slam_frame;
            h_valid_keypoints_num_previous = h_valid_keypoints_num;

            std::cout << "Finished work GPU thread " << slam_frames_id
                      << " duration " << duration.count()
                      << " duration_rotation " << duration_rotation.count()
                      << " h_valid_keypoints_num " << h_valid_keypoints_num
                      //   << " theta.x " << theta.x
                      //   << " theta.y " << theta.y
                      //   << " theta.z " << theta.z
                      //   << " keypoints_num " << keypoints_num
                      << std::endl;
        }

        checkCudaErrors(cudaFree(d_rgb_image));
        checkCudaErrors(cudaFree(d_gray_image));
        checkCudaErrors(cudaFree(d_aligned_out));
        checkCudaErrors(cudaFree(d_depth_in));
        // checkCudaErrors(cudaFree(d_keypoints_exist));
        // checkCudaErrors(cudaFree(d_keypoints_response));
        printf("---mark---");
        checkCudaErrors(cudaFree(d_keypoints_angle));
        checkCudaErrors(cudaFree(d_descriptors_tmp));
        checkCudaErrors(cudaFree(d_descriptors));

        std::cout << "Stopped GPU thread " << slam_frames_id << std::endl;
    }

}