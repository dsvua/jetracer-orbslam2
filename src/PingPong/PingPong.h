#ifndef JETRACER_PING_PONG_THREAD_H
#define JETRACER_PING_PONG_THREAD_H

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
    class PingPong : public EventsThread
    {
    public:
        PingPong(const std::string threadName, context_t *ctx);
        // ~PingPong();

    private:
        void TimerThread();
        void handleEvent(pEvent event);

        context_t *_ctx;
        std::mutex m_mutex_subscribers;
        std::atomic<bool> m_timerExit;
        std::thread *timerThread;
    };
} // namespace Jetracer

#endif // JETRACER_PING_PONG_THREAD_H
