
// #include <opencv2/opencv.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/imgproc.hpp>
#include "types.h"
#include "realsense_camera.h"
#include "image_pipeline.h"

int main(int argc, char * argv[]) {
    // cv::String config_file_name(argv[1]);
    Jetracer::context_t ctx;
    // Jetracer::Configuration jetracer_configuration(config_file_name, &ctx); // not implemented yet

    // start camera capturing
    Jetracer::realsenseD435iThread jetracer_depth_camera(&ctx);
    jetracer_depth_camera.initialize();
    jetracer_depth_camera.waitRunning();

    // start image processing pipeline
    Jetracer::imagePipelineThread jetracer_image_pipeline(&ctx);
    jetracer_image_pipeline.initialize();
    jetracer_image_pipeline.waitRunning();
}

// image ignore area: 60px on left and 50px on bottom. 0,0 is top left corner


// fit plane to points: plane formula ax + by + c = z
// Python example
// # do fit
// tmp_A = []
// tmp_b = []
// for i in range(len(xs)):
//     tmp_A.append([xs[i], ys[i], 1])
//     tmp_b.append(zs[i])
// b = np.matrix(tmp_b).T
// A = np.matrix(tmp_A)
// fit = (A.T * A).I * A.T * b
