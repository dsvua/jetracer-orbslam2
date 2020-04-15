#include "Thread.h"
#include "constants.h"
#include "types.h"
#include <librealsense2/rs.hpp>

#ifndef JETRACER_REALSENSE_D435I_THREAD_H
#define JETRACER_REALSENSE_D435I_THREAD_H


namespace Jetracer {

    class realsenseD435iThread : public Thread {
    public:
        explicit realsenseD435iThread(context_t * ctx)
                                    : _ctx(ctx)
        {
        }
        ~realsenseD435iThread() {}
        // rs2::frame_queue depth_queue(CAPACITY);
        // rs2::frame_queue left_ir_queue(CAPACITY);
        // rs2::frame_queue right_ir_queue(CAPACITY);
    private:
        virtual bool threadInitialize();
        virtual bool threadExecute();
        virtual bool threadShutdown();

        context_t * _ctx;
        rs2::config _config;
        rs2::pipeline _pipeline;
        rs2::pipeline_profile _pipeline_profile;
    }
} // namespace Jetracer

#endif // JETRACER_REALSENSE_D435I_THREAD_H
