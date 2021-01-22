#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "popt_pp.h"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
    char* img_filename;
    char* calib_file;
    char* leftout_filename;
    char* rightout_filename;
    int image_width=1280;
    int image_height=720;
    Size image_size = Size(image_width, image_height);
    size_t imageSize = image_width*image_height*sizeof(uchar);

    static struct poptOption options[] = {
        { "img_filename",'l',POPT_ARG_STRING,&img_filename,0,"Left imgage path","STR" },
        { "calib_file",'c',POPT_ARG_STRING,&calib_file,0,"Stereo calibration file","STR" },
        { "leftout_filename",'L',POPT_ARG_STRING,&leftout_filename,0,"Left undistorted imgage path","STR" },
        { "rightout_filename",'R',POPT_ARG_STRING,&rightout_filename,0,"Right undistorted image path","STR" },
    POPT_AUTOHELP
        { NULL, 0, 0, NULL, 0, NULL, NULL }
    };

    POpt popt(NULL, argc, argv, options, 0);
    int c;
    while((c = popt.getNextOpt()) >= 0) {}

    Mat left_R, right_R, left_P, right_P, Q;
    Mat left_K, right_K, R;
    Vec3d T;
    Mat left_D, right_D;

    FILE *image_file = fopen(img_filename, "rb");
    Mat left_image(image_size, CV_8UC1);
    Mat right_image(image_size, CV_8UC1);
    if (image_file){
        fread ( (uchar*)left_image.data, sizeof(uchar), imageSize, image_file );
        fread ( (uchar*)right_image.data, sizeof(uchar), imageSize, image_file );
    }
    fclose(image_file);
    // gamma correction for images
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    float image_gamma = 0.45;
    for( int i = 0; i < 256; ++i)
    p[i] = saturate_cast<uchar>(pow(i / 255.0, image_gamma) * 255.0);
    Mat  left_image_gamma =  left_image.clone();
    Mat right_image_gamma = right_image.clone();
    // LUT(left_image, lookUpTable, left_image_gamma);
    // LUT(right_image, lookUpTable, right_image_gamma);

    cv::Mat lmapx, lmapy, rmapx, rmapy;
    cv::Mat left_imgU, right_imgU;

    cv::FileStorage fs1(calib_file, cv::FileStorage::READ);
    fs1["left_K"] >> left_K;
    fs1["right_K"] >> right_K;
    fs1["left_D"] >> left_D;
    fs1["right_D"] >> right_D;
    fs1["R"] >> R;
    fs1["T"] >> T;

    fs1["left_R"] >> left_R;
    fs1["right_R"] >> right_R;
    fs1["left_P"] >> left_P;
    fs1["right_P"] >> right_P;
    fs1["Q"] >> Q;
    fs1["lmapx"] >> lmapx;
    fs1["lmapy"] >> lmapy;
    fs1["rmapx"] >> rmapx;
    fs1["rmapy"] >> rmapy;


    // cv::initUndistortRectifyMap(left_K, left_D, left_R, left_P, left_image_gamma.size(), CV_32F, lmapx, lmapy);
    // cv::initUndistortRectifyMap(right_K, right_D, right_R, right_P, right_image_gamma.size(), CV_32F, rmapx, rmapy);
    cv::remap(left_image_gamma, left_imgU, lmapx, lmapy, cv::INTER_LINEAR);
    cv::remap(right_image_gamma, right_imgU, rmapx, rmapy, cv::INTER_LINEAR);

    imwrite(leftout_filename, left_imgU);
    imwrite(rightout_filename, right_imgU);

    return 0;
}
