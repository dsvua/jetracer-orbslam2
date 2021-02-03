#ifndef JETRACER_SLAM_GPU_PIPELINE_THREAD_H
#define JETRACER_SLAM_GPU_PIPELINE_THREAD_H

#include <iostream>

#include "../EventsThread.h"
#include "../Context.h"
#include "../Events/BaseEvent.h"
#include "../Events/EventTypes.h"
#include <mutex>
#include <atomic>
#include <thread>

namespace Jetracer
{

    class SlamGpuPipeline : public EventsThread
    {
    public:
        SlamGpuPipeline(const std::string threadName, context_t *ctx);
        // ~SlamGpuPipeline();

    private:
        void handleEvent(pEvent event);

        context_t *_ctx;
        std::mutex m_mutex_subscribers;
        };
} // namespace Jetracer

#endif // JETRACER_SLAM_GPU_PIPELINE_THREAD_H
