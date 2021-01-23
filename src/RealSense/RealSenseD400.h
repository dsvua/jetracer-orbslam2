#ifndef JETRACER_REALSENSE_D400_THREAD_H
#define JETRACER_REALSENSE_D400_THREAD_H

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

    // This class is an example of plain message sending/receiving
    // It is used for testing messaging facility
    class RealSenseD400 : public EventsThread
    {
    public:
        RealSenseD400(const std::string threadName, context_t *ctx);
        // ~RealSenseD400();

    private:
        void TimerThread();
        void handleEvent(pEvent event);

        context_t *_ctx;
        std::mutex m_mutex_subscribers;
    };
} // namespace Jetracer

#endif // JETRACER_REALSENSE_D400_THREAD_H
