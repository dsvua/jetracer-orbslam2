#include "Thread.h"

#ifndef JETRACER_REALSENSE_D435I_THREAD_H
#define JETRACER_REALSENSE_D435I_THREAD_H


namespace Jetracer {

    class realsenseD435iThread : public Thread {
    public:
        explicit realsenseD435iThread(int width, int height, int fps)
                                    : _width = width
                                    , _height = height
                                    , _fps = fps
        {
        }
        ~realsenseD435iThread() {}
    private:
        virtual bool threadInitialize();
        virtual bool threadExecute();
        virtual bool threadShutdown();

        // camera settings
        const auto CAPACITY = 5; // allow max latency of 5 frames

        int _width;
        int _height;
        int _fps;
        int emitter_mode, frame_count;
        rs2::pipeline pipeline;
        rs2::depth_sensor depth_sensor;
        rs2::frame_queue depth_queue(CAPACITY);
        rs2::frame_queue left_ir_queue(CAPACITY);
        rs2::frame_queue right_ir_queue(CAPACITY);

    }
} // namespace Jetracer

#endif // JETRACER_REALSENSE_D435I_THREAD_H
