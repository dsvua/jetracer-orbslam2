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
    const string inputSettingsFile = argc > 1 ? argv[1] : "default.xml";
    FileStorage left_fs_config("left_" + inputSettingsFile, FileStorage::READ); // Read the settings
    FileStorage right_fs_config("right_" + inputSettingsFile, FileStorage::READ); // Read the settings
    if (!left_fs_config.isOpened() || !right_fs_config.isOpened())
    {
        cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
        return -1;
    }
    Mat left_map1, left_map2, right_map1, right_map2;
    int image_width, image_height, imageSize;
    left_fs_config["image_width"] >> image_width;
    left_fs_config["image_height"] >> image_height;
    left_fs_config["map1"] >> left_map1;
    left_fs_config["map2"] >> left_map2;
    right_fs_config["map1"] >> right_map1;
    right_fs_config["map2"] >> right_map2;

    imageSize = image_width * image_height;
    Size image_size = Size(image_width, image_height);


    const string inputImageFile = argc > 2 ? argv[2] : "picture.png";
    FILE *image_file = fopen(inputImageFile.c_str(), "rb");
    Mat left_image(image_size, CV_8UC1);
    Mat right_image(image_size, CV_8UC1);
    if (image_file){
        fread ((uchar*)left_image.data, sizeof(uchar), imageSize, image_file);
        fread ((uchar*)right_image.data, sizeof(uchar), imageSize, image_file);
    }
    fclose(image_file);

    cout << inputImageFile << " file processed " << image_width << "x" << image_height << endl;
    cout << left_image.size() << right_image.size() << endl;
    cout << left_map1.size() << left_map2.size() << endl;
    cout << right_map1.size() << right_map2.size() << endl;

    Mat left_dst_image, right_dst_image;
    cout << "remapping left image" << endl;
    remap(left_image, left_dst_image, left_map1, left_map2, INTER_LINEAR);
    cout << "remapping right image" << endl;
    remap(right_image, right_dst_image, right_map1, right_map2, INTER_LINEAR);
    imshow("left_Image View", left_dst_image);
    imshow("right_Image View", right_dst_image);
    waitKey(0);

    left_fs_config.release();                                         // close Settings file
    right_fs_config.release();                                         // close Settings file    

}