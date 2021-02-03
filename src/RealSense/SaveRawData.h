#ifndef JETRACER_TEMPLATE_THREAD_H
#define JETRACER_TEMPLATE_THREAD_H

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

    class SaveRawData : public EventsThread
    {
    public:
        SaveRawData(const std::string threadName, context_t *ctx);
        // ~SaveRawData();

    private:
        void handleEvent(pEvent event);

        context_t *_ctx;
        std::mutex m_mutex_subscribers;
    };
} // namespace Jetracer

#endif // JETRACER_TEMPLATE_THREAD_H
