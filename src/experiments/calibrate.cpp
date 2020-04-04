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
        for(auto&& file: globVector((string)(dir + "/*.bin"), GLOB_TILDE)){
            // printf("Adding file %s to files\n", file.c_str());
            std::cout << file << std::endl;
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

    string calib_file;
    string img_dir;

    calib_file = "calibration_images.yaml";
    img_dir = "~/Downloads/images_temp/bin";

    Mat  left_image_gamma;
    Mat right_image_gamma;


    vector<string> files = getPicturesFileNames(img_dir);
    std::string s_ops = "rb";
    cout << "Calibration files " << endl;
    for(auto&& filename: files){
        cout << "Processing: " << filename << endl;
        // const char *c_ops = s_ops.c_str();
        FILE *image_file = fopen(filename.c_str(), s_ops.c_str());
        // cout << filename << " opened "<< endl;
        Mat left_image(image_size, CV_8UC1);
        Mat right_image(image_size, CV_8UC1);
        if (image_file){
            // cout << "Reading image " << filename << endl;
            fread ( (uchar*)left_image.data, sizeof(uchar), imageSize, image_file );
            fread ( (uchar*)right_image.data, sizeof(uchar), imageSize, image_file );
            // cout << "Image readed" << filename << endl;
        }
        fclose(image_file);
        // gamma correction for images
        Mat lookUpTable(1, 256, CV_8U);
        uchar* p = lookUpTable.ptr();
        float image_gamma = 0.45;
        for( int i = 0; i < 256; ++i)
            p[i] = saturate_cast<uchar>(pow(i / 255.0, image_gamma) * 255.0);
        left_image_gamma  = left_image.clone();
        right_image_gamma = right_image.clone();
        // LUT(left_image, lookUpTable, left_image_gamma);
        // LUT(right_image, lookUpTable, right_image_gamma);

        imwrite( (filename + "left_cv_gamma.png"), left_image_gamma );
        imwrite( (filename + "right_cv_gamma.png"), right_image_gamma );

        bool left_found = false, right_found = false;
        left_found = cv::findChessboardCorners(left_image_gamma, board_size, left_corners,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        // cout << "left image cv::findChessboardCorners" << endl;
        right_found = cv::findChessboardCorners(right_image_gamma, board_size, right_corners,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        // cout << "right image cv::findChessboardCorners" << endl;
        if(!left_found || !right_found){
            cout << "Chessboard find error for file " << left_found << "**" << right_found << " " << filename << endl;
            continue;
        }

        cv::cornerSubPix(left_image_gamma, left_corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.0001 ));
        // cv::drawChessboardCorners(left_image_gamma, board_size, left_corners, left_found);
        cv::cornerSubPix(right_image_gamma, right_corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.0001 ));
        // cv::drawChessboardCorners(right_image_gamma, board_size, right_corners, right_found);
        // cout << "Corners are searched" << endl;

        vector< Point3f > obj;
        for (int i = 0; i < board_height; i++)
            for (int j = 0; j < board_width; j++)
                obj.push_back(Point3f((float)j * square_size, (float)i * square_size, 0));

        cout << filename << ". Found corners!" << left_corners.size() << " " << obj.size() << endl;
        left_imagePoints.push_back(left_corners);
        right_imagePoints.push_back(right_corners);
        for (int i = 0; i < left_imagePoints.size(); i++) {
            vector< Point2f > left_v, right_v;
            for (int j = 0; j < left_imagePoints[i].size(); j++) {
                left_v.push_back(Point2f((double)left_imagePoints[i][j].x, (double)left_imagePoints[i][j].y));
                right_v.push_back(Point2f((double)right_imagePoints[i][j].x, (double)right_imagePoints[i][j].y));
            }
            left_img_points.push_back(left_v);
            right_img_points.push_back(right_v);
            object_points.push_back(obj);
            // cout << "object_points: " << object_points.size() << " left_img_points: " << left_img_points.size() 
            //         << " right_img_points: " << right_img_points.size() << endl;
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

        cout << "object_points: " << object_points.size() << " left_img_points: " << left_img_points.size() 
                    << " right_img_points: " << right_img_points.size() << endl;
        
        cout << "calibrating left camera" << endl;
        calibrateCamera(object_points, left_img_points, image_size, left_K, left_D, left_rvecs, left_tvecs, flag);
        cout << "left camera calibrated, working on right camera" << endl;
        calibrateCamera(object_points, right_img_points, image_size, right_K, right_D, right_rvecs, right_tvecs, flag);
        cout << "right camera calibrated, working on calibration error" << endl;

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
        cv::Mat lmapx, lmapy, rmapx, rmapy;
        cv::Mat left_imgU, right_imgU;
        cv::initUndistortRectifyMap(left_K, left_D, left_R, left_P, left_image_gamma.size(), CV_32F, lmapx, lmapy);
        cv::initUndistortRectifyMap(right_K, right_D, right_R, right_P, right_image_gamma.size(), CV_32F, rmapx, rmapy);

        fs_config << "lmapx" << lmapx;
        fs_config << "lmapy" << lmapy;
        fs_config << "rmapx" << rmapx;
        fs_config << "rmapy" << rmapy;

        cout << "Done with initUndistortRectifyMap" << endl;

    } else {
        printf("No images - no calibration!\n");
    }
    return 0;

}
