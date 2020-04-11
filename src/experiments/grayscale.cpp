#include "cudaYUVtoGray.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdlib.h>
// #include <cuda_runtime_api.h>
#include <cuda.h>
using namespace cv;

int main(void)
{

    // string filename = "/home/serhiy/Pictures/calibrate/";

    Mat color_im = imread("/home/serhiy/Pictures/calibrate/1_output195.jpg");

    cv::namedWindow("Detected", cv::WINDOW_AUTOSIZE);
    cv::imshow("Detected", color_im);
    cv::waitKey(0);

    Mat color_im_f;
    color_im.convertTo(color_im_f, CV_8U, 1/255.0);
    // color_im.convertTo(color_im_f, CV_32FC3);

    // cv::namedWindow("Detected", cv::WINDOW_AUTOSIZE);
    cv::imshow("Detected", color_im_f);
    cv::waitKey(0);



    Mat grayscale_im;
    Mat image_YUV_I420;

    cvtColor(color_im_f, image_YUV_I420, COLOR_BGRA2YUV_I420);
    cv::imshow("Detected", color_im_f);
    cv::waitKey(0);

    cudaYUYVToGray(image_YUV_I420, grayscale_im, image_YUV_I420.rows, image_YUV_I420.cols);
    cv::imshow("Detected", grayscale_im);
    cv::waitKey(0);

    std::cout << std::to_string(color_im.type()) << std::endl;

}
