#include "TemplateEventsThread.h"

#include <memory>
#include <chrono>
#include <iostream>

// using namespace std;

namespace Jetracer
{
    TemplateEventsThread::TemplateEventsThread(const std::string threadName, context_t *ctx) : EventsThread(threadName), _ctx(ctx)
    {
        auto pushEventCallback = [this](pEvent event) -> bool {
            this->pushEvent(event);
            return true;
        };

        _ctx->subscribeForEvent(EventType::event_ping, threadName, pushEventCallback);
        _ctx->subscribeForEvent(EventType::event_pong, threadName, pushEventCallback);

        std::cout << "TemplateEventsThread is initialized" << std::endl;
    }

    void TemplateEventsThread::handleEvent(pEvent event)
    {

        switch (event->event_type)
        {

        case EventType::event_stop_thread:
        {
            break;
        }

        default:
        {
            // std::cout << "Got unknown message of type " << event->event_type << std::endl;
            break;
        }
        }
    }

} // namespace Jetracer