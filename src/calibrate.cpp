#include <string>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <glob.h>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// #include "popt_pp.h"

using std::vector;
using namespace std;
using namespace cv;

vector<string> globVector(const string& pattern, int flags){
    glob_t glob_result;
    glob(pattern.c_str(),flags,NULL,&glob_result);
    vector<string> files;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        files.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}

vector<string> getPicturesFileNames(string path){
    vector<string> dirs = globVector((string)(path + "/*"), GLOB_TILDE | GLOB_ONLYDIR);
    vector<string> files;

    for(auto&& dir: dirs){
        for(auto&& file: globVector((string)(dir + "/*"), GLOB_TILDE)){
            // std::cout << file << std::endl;
            files.push_back(file);
        }
    }
    return files;
}

double computeReprojectionErrors(const vector< vector< Point3f > >& objectPoints,
                                 const vector< vector< Point2f > >& imagePoints,
                                 const vector< Mat >& rvecs, const vector< Mat >& tvecs,
                                 const Mat& cameraMatrix , const Mat& distCoeffs) {
  vector< Point2f > imagePoints2;
  int i, totalPoints = 0;
  double totalErr = 0, err;
  vector< float > perViewErrors;
  perViewErrors.resize(objectPoints.size());

  for (i = 0; i < (int)objectPoints.size(); ++i) {
    projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
                  distCoeffs, imagePoints2);
    err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);
    int n = (int)objectPoints[i].size();
    perViewErrors[i] = (float) std::sqrt(err*err/n);
    totalErr += err*err;
    totalPoints += n;
  }
  return std::sqrt(totalErr/totalPoints);
}

int main(int argc, char const *argv[])
{
    vector< vector< Point3f > > object_points;
    vector< vector< Point2f > > left_imagePoints, right_imagePoints;
    vector< Point2f >           left_corners,     right_corners;
    vector< vector< Point2f > > left_img_points,  right_img_points;

    int board_height = 6;
    int board_width = 9;
    int image_width=1280;
    int image_height=720;
    int square_size = 1;
    Size board_size = Size(board_width, board_height);
    Size image_size = Size(image_width, image_height);
    int board_n = board_width * board_height;
    size_t imageSize = image_width*image_height*sizeof(uint8_t);
    uint8_t *left_image, *right_image;

    string calib_file;
    string img_dir;

    // static struct poptOption options[] = {
    //     { "calib_file",'c',POPT_ARG_STRING,&calib_file,0,"Left camera calibration","STR" },
    //     { "img_dir",'i',POPT_ARG_STRING,&img_dir,0,"Directory containing left images","STR" },
    //     POPT_AUTOHELP
    //     { NULL, NULL }
    // };
    // POpt popt(NULL, argc, argv, options, 0);

    calib_file = "calibration.yaml";
    img_dir = "images_bin";

    vector<string> files = getPicturesFileNames(calib_file);
    cout << "Calibration files " << endl;
    for(auto&& filename: files){
        cout << filename << endl;
        std::string s_ops = "rb";
        // const char *c_ops = s_ops.c_str();
        FILE *file = fopen(filename.c_str(), s_ops.c_str());
        if (file){
            left_image = (uint8_t*)malloc(imageSize);
            right_image = (uint8_t*)malloc(imageSize);
            fread ( left_image, sizeof(uint8_t), imageSize, file );
            fread ( right_image, sizeof(uint8_t), imageSize, file );
        }
        fclose(file);
        // Mat left_image(image_height, image_width, CV_8UC1, &left_image);
        // Mat right_image(image_height, image_width, CV_8UC1, &right_image);
        Mat left_image(image_size, CV_8UC1, &left_image);
        Mat right_image(image_size, CV_8UC1, &right_image);

        bool left_found = false, right_found = false;
        left_found = cv::findChessboardCorners(left_image, board_size, left_corners,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        right_found = cv::findChessboardCorners(right_image, board_size, right_corners,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        if(left_found && right_found){
            cv::cornerSubPix(left_image, left_corners, cv::Size(5, 5), cv::Size(-1, -1),
                    cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            // cv::drawChessboardCorners(left_image, board_size, left_corners, left_found);
            cv::cornerSubPix(right_image, right_corners, cv::Size(5, 5), cv::Size(-1, -1),
                    cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            // cv::drawChessboardCorners(right_image, board_size, right_corners, right_found);
        }

        vector< Point3f > obj;
        for (int i = 0; i < board_height; i++)
            for (int j = 0; j < board_width; j++)
                obj.push_back(Point3f((float)j * square_size, (float)i * square_size, 0));

        if (left_found && right_found) {
            cout << filename << ". Found corners!" << endl;
            left_imagePoints.push_back(left_corners);
            right_imagePoints.push_back(right_corners);
            object_points.push_back(obj);
        }
        for (int i = 0; i < left_imagePoints.size(); i++) {
            vector< Point2f > left_v, right_v;
            for (int j = 0; j < left_imagePoints[i].size(); j++) {
                left_v.push_back(Point2f((double)left_imagePoints[i][j].x, (double)left_imagePoints[i][j].y));
                right_v.push_back(Point2f((double)right_imagePoints[i][j].x, (double)right_imagePoints[i][j].y));
            }
            left_img_points.push_back(left_v);
            right_img_points.push_back(right_v);
        }

    }

    if (files.size() > 0) {
        // calibrating each camera
        printf("Starting intristic Calibration\n");
        Mat left_K, right_K;
        Mat left_D, right_D;
        vector< Mat > left_rvecs, left_tvecs, right_rvecs, right_tvecs;
        int flag = 0;
        flag |= CV_CALIB_FIX_K4;
        flag |= CV_CALIB_FIX_K5;
        calibrateCamera(object_points, left_img_points, image_size, left_K, left_D, left_rvecs, left_tvecs, flag);
        calibrateCamera(object_points, right_img_points, image_size, right_K, right_D, right_rvecs, right_tvecs, flag);

        cout << "Calibration error left: " << computeReprojectionErrors(object_points, left_img_points,
                    left_rvecs, left_tvecs, left_K, left_D) << endl;
        cout << "Calibration error right: " << computeReprojectionErrors(object_points, right_img_points,
                    right_rvecs, right_tvecs, right_K, right_D) << endl;

        FileStorage fs_config(calib_file, FileStorage::WRITE);
        fs_config << "left_K" << left_K;
        fs_config << "left_D" << left_D;
        fs_config << "right_K" << right_K;
        fs_config << "right_D" << right_D;
        fs_config << "board_width" << board_width;
        fs_config << "board_height" << board_height;
        fs_config << "square_size" << square_size;
        printf("Done intristic Calibration\n");


        // stereo calibration

        printf("Starting Stereo Calibration\n");
        Mat R, F, E;
        Vec3d T;
        flag = 0;
        flag |= CV_CALIB_FIX_INTRINSIC;

        stereoCalibrate(object_points, left_img_points, right_img_points, left_K, left_D, right_K,
                    right_D, image_size, R, T, E, F);

        fs_config << "R" << R;
        fs_config << "T" << T;
        fs_config << "E" << E;
        fs_config << "F" << F;

        printf("Done Calibration\n");

        printf("Starting Rectification\n");

        cv::Mat left_R, right_R, left_P, right_P, Q;
        stereoRectify(left_K, left_D, right_K, right_D, image_size, R, T,
                    left_R, right_R, left_P, right_P, Q);

        fs_config << "left_R" << left_R;
        fs_config << "right_R" << right_R;
        fs_config << "left_P" << left_P;
        fs_config << "right_P" << right_P;
        fs_config << "Q" << Q;

        printf("Done Rectification\n");
    } else {
        printf("No images - no calibration!\n");
    }
    return 0;

}
