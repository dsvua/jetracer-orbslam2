#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;


int main(int argc, char* argv[])
{

    //! [file_read]
    // Settings s;
    const string inputSettingsFile = argc > 1 ? argv[1] : "default.xml";
    FileStorage left_fs_config("left_" + inputSettingsFile, FileStorage::READ); // Read the settings
    FileStorage right_fs_config("right_" + inputSettingsFile, FileStorage::READ); // Read the settings
    if (!left_fs_config.isOpened() || !right_fs_config.isOpened())
    {
        cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
        return -1;
    }
    int image_width, image_height, board_width, board_height, nr_of_frames;
    float square_size;
    Mat left_camera_matrix, left_distortion_coefficients, right_camera_matrix, right_distortion_coefficients;
    Mat R, T, left_image_points, right_image_points;
    // cv::Vec3d T;
    vector< vector< Point3f > > object_points;
    vector< vector< Point2f > > left_imagePoints, right_imagePoints;

    // left_distortion_coefficients = Mat::zeros(4, 1, CV_64F);
    // right_distortion_coefficients = Mat::zeros(4, 1, CV_64F);

    cout << "image_width" << endl;
    left_fs_config["image_width"] >> image_width;
    cout << "image_height" << endl;
    left_fs_config["image_height"] >> image_height;
    cout << "board_width" << endl;
    left_fs_config["board_width"] >> board_width;
    cout << "board_height" << endl;
    left_fs_config["board_height"] >> board_height;
    cout << "nr_of_frames" << endl;
    left_fs_config["nr_of_frames"] >> nr_of_frames;
    cout << "square_size" << endl;
    left_fs_config["square_size"] >> square_size;

    cout << "\nLeft camera" << endl;
    cout << "camera_matrix" << endl;
    left_fs_config["camera_matrix"] >> left_camera_matrix;
    cout << "image_points" << endl;
    left_fs_config["image_points"] >> left_image_points;
    cout << "distortion_coefficients" << endl;
    left_fs_config["distortion_coefficients"] >> left_distortion_coefficients;

    cout << "\nRight camera" << endl;
    cout << "camera_matrix" << endl;
    right_fs_config["camera_matrix"] >> right_camera_matrix;
    cout << "image_points" << endl;
    right_fs_config["image_points"] >> right_image_points;
    cout << "distortion_coefficients" << endl;
    right_fs_config["distortion_coefficients"] >> right_distortion_coefficients;
    cout << "Config is read" << endl;

    left_fs_config.release();                                         // close Settings file
    right_fs_config.release();                                         // close Settings file

    Mat left_newCamMat, right_newCamMat;
    Mat left_view, left_rview, left_map1, left_map2;
    Mat right_view, right_rview, right_map1, right_map2;

    fisheye::estimateNewCameraMatrixForUndistortRectify(left_camera_matrix, left_distortion_coefficients,
            Size(image_width, image_height), Matx33d::eye(), left_newCamMat, 0);
    fisheye::estimateNewCameraMatrixForUndistortRectify(right_camera_matrix, right_distortion_coefficients,
            Size(image_width, image_height), Matx33d::eye(), right_newCamMat, 0);

    fisheye::initUndistortRectifyMap(left_camera_matrix, left_distortion_coefficients, Matx33d::eye(),
            left_newCamMat, Size(image_width, image_height), CV_16SC2, left_map1, left_map2);
    fisheye::initUndistortRectifyMap(right_camera_matrix, right_distortion_coefficients, Matx33d::eye(),
            right_newCamMat, Size(image_width, image_height), CV_16SC2, right_map1, right_map2);

    // remap(src_image, dst_image, map1, map2, INTER_LINEAR);
    // imshow("Image View", dst_image);

    cout << left_map1.size() << left_map2.size() << endl;
    cout << right_map1.size() << right_map2.size() << endl;

    FileStorage left_fs_config_w("left_remap_" + inputSettingsFile, FileStorage::WRITE);
    left_fs_config_w << "image_width" << image_width;
    left_fs_config_w << "image_height" << image_height;
    // left_fs_config_w << "board_width" << board_width;
    // left_fs_config_w << "board_height" << board_height;
    // left_fs_config_w << "nr_of_frames" << nr_of_frames;
    // left_fs_config_w << "square_size" << square_size;

    // left_fs_config_w << "camera_matrix" << left_camera_matrix;
    // left_fs_config_w << "image_points" << left_image_points;
    // left_fs_config_w << "distortion_coefficients" <<left_distortion_coefficients;
    left_fs_config_w << "map1" << left_map1;
    left_fs_config_w << "map2" << left_map2;
    left_fs_config_w.release();                                         // close Settings file

    FileStorage right_fs_config_w("right_remap_" + inputSettingsFile, FileStorage::WRITE);
    right_fs_config_w << "image_width" << image_width;
    right_fs_config_w << "image_height" << image_height;
    // right_fs_config_w << "board_width" << board_width;
    // right_fs_config_w << "board_height" << board_height;
    // right_fs_config_w << "nr_of_frames" << nr_of_frames;
    // right_fs_config_w << "square_size" << square_size;
    // right_fs_config_w << "camera_matrix" << right_camera_matrix;
    // right_fs_config_w << "image_points" << right_image_points;
    // right_fs_config_w << "distortion_coefficients" << right_distortion_coefficients;
    right_fs_config_w << "map1" << right_map1;
    right_fs_config_w << "map2" << right_map2;

    right_fs_config_w.release();                                         // close Settings file
}
